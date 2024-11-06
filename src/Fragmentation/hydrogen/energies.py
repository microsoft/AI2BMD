import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from Fragmentation.hydrogen.topology import ProteinData


@torch.jit.script
def _get_bond_energy_jit(coord, bond_idx_src, bond_idx_dst, bond_force_constant, bond_idx, bond_equil_value, bond_batch):
    dist = torch.norm(coord[bond_idx_src] - coord[bond_idx_dst], dim=-1)
    energy = bond_force_constant[bond_idx] * (dist - bond_equil_value[bond_idx]).square()
    return scatter_add(energy, bond_batch, dim=0) * 0.5


@torch.jit.script
def _get_angle_energy_jit(coord, angles_i, angles_j, angles_k, angle_force_constant, angle_idx, angle_equil_value, angle_batch):
    v0 = coord[angles_i] - coord[angles_j]
    v1 = coord[angles_k] - coord[angles_j]
    y = torch.norm(torch.cross(v0, v1, dim=-1), dim=-1)
    x = torch.einsum('ij,ij->i', v0, v1)
    angle = torch.atan2(y, x)

    energy = angle_force_constant[angle_idx] * (angle - angle_equil_value[angle_idx]).square()
    return scatter_add(energy, angle_batch, dim=0) * 0.5


@torch.jit.script
def _get_dihedral_energy_jit(coord, dih_i, dih_j, dih_k, dih_l, dih_force_constant, dih_idx, dih_periodicity, dih_phase, dih_batch):
    p0, p1 = coord[dih_i], coord[dih_j]
    p2, p3 = coord[dih_k], coord[dih_l]
    v0, v1, v2 = p1 - p2, p1 - p0, p3 - p2
    n1 = F.normalize(torch.cross(v1, v0, dim=-1), dim=-1)
    n2 = F.normalize(torch.cross(v0, v2, dim=-1), dim=-1)
    m1 = torch.cross(n1, F.normalize(v0, dim=-1), dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    dihedral = torch.atan2(y, x)
    # TODO: There is discrepancy between the Amber manual and the PDF.
    #  Which one is correct?
    energy = dih_force_constant[dih_idx] * (
        1 + torch.cos(dih_periodicity[dih_idx] * dihedral - dih_phase[dih_idx])
    )
    return scatter_add(energy, dih_batch, dim=0) * 0.5


@torch.jit.script
def _get_vdw_energy_jit(lj_acoef, lj_bcoef, lj_idx, nonbonded_batch, scnb_scale_factor, dist):
    # TODO: Amber discount the 1-4 VDW interactions by `scnb_scale_factor`.
    #  Implement the discounting.
    r6 = torch.pow(dist, 6)
    r12 = torch.square(r6)
    energy = lj_acoef[lj_idx] / r12 - lj_bcoef[lj_idx] / r6
    return scatter_add(energy, nonbonded_batch, dim=0) / scnb_scale_factor


@torch.jit.script
def _get_elec_energy_jit(charge, nonbonded_idx_src, nonbonded_idx_dst, nonbonded_batch, scee_scale_factor, dist):
    # TODO: Amber discount the 1-4 electrostatic
    #  interactions by `scee_scale_factor`. Implement the discounting.
    energy = charge[nonbonded_idx_src] * charge[nonbonded_idx_dst] / dist
    return scatter_add(energy, nonbonded_batch, dim=0) / scee_scale_factor


class HydrogenOptimizer:
    """
    Hydrogen coordinate optimizer.
    It contains an energy calculator and an optimizer.
    Amber potential energy is calculated including the bond, angle,
     and dihedral energies.
    Non-bonded energies include the van der Waals energy
    and the electrostatic energy.
    Torch's `LBFGS` implementation is used as the optimizer.
    """

    def __init__(
        self, max_iter=5, scnb_scale_factor=1.2, scee_scale_factor=2.0
    ):
        self.max_iter = max_iter
        self.scnb_scale_factor = scnb_scale_factor
        self.scee_scale_factor = scee_scale_factor

    def get_bond_energy(self, batch: ProteinData):
        r"""
        Bond energy are calculated according to the equation

        .. math::
            E_{bond} = \frac{1}{2} k (r - r_{eq})^2

        :param batch: batch of `ProteinData`
        :return: graph-level bond energy
        """
        return _get_bond_energy_jit(
            batch.pos,
            batch.bonds_atom_idx_src,
            batch.bonds_atom_idx_dst,
            batch.bond_force_constant,
            batch.bond_idx,
            batch.bond_equil_value,
            batch.bond_batch,
        )

    def get_angle_energy(self, batch: ProteinData):
        r"""
        Angle energy are calculated according to the equation

        .. math::
            E_{angle} = \frac{1}{2} k_\theta (\theta - \theta_{eq})^2

        :param batch: batch of `ProteinData`
        :return: graph-level angle energy
        """
        return _get_angle_energy_jit(
            batch.pos,
            batch.angles_atom_idx_i,
            batch.angles_atom_idx_j,
            batch.angles_atom_idx_k,
            batch.angle_force_constant,
            batch.angle_idx,
            batch.angle_equil_value,
            batch.angle_batch,
        )

    def get_dihedral_energy(self, batch: ProteinData):
        r"""
        Dihedral (torsion) energy are calculated according to the equation

        .. math::
            E_{dihedral} = \frac{1}{2} k_{tor} (1 + \cos(n \phi - \psi))

        :param batch: batch of `ProteinData`
        :return: graph-level angle energy
        """

        return _get_dihedral_energy_jit(
            batch.pos,
            batch.dihedrals_atom_idx_i,
            batch.dihedrals_atom_idx_j,
            batch.dihedrals_atom_idx_k,
            batch.dihedrals_atom_idx_l,
            batch.dihedral_force_constant,
            batch.dihedral_idx,
            batch.dihedral_periodicity,
            batch.dihedral_phase,
            batch.dihedral_batch,
        )

    def get_vdw_energy(self, batch: ProteinData, dist):
        r"""
        Lennard-Jones potential for van der Waals interactions.

        .. math::
            E_{vdw} = \frac{a_{ij}}{r_{ij}^{12}} - \frac{b_{ij}}{r_{ij}^{6}}

        :param batch: batch of `ProteinData`
        :param dist: distance between atoms
        :return: graph-level electrostatic energy
        """

        return _get_vdw_energy_jit(
            batch.lennard_jones_acoef,
            batch.lennard_jones_bcoef,
            batch.lj_idx,
            batch.nonbonded_batch,
            self.scnb_scale_factor,
            dist,
        )

    def get_elec_energy(self, batch: ProteinData, dist):
        r"""
        Coulomb potential for electrostatic interactions.

        .. math::
            E_{elec} = \frac{q_i q_j}{4 \pi \epsilon_0 r_{ij}}

        :param batch: batch of `ProteinData`
        :param dist: distance between atoms
        :return: graph-level electrostatic energy
        """
        return _get_elec_energy_jit(
            batch.charge,
            batch.nonbonded_atom_idx_src,
            batch.nonbonded_atom_idx_dst,
            batch.nonbonded_batch,
            self.scee_scale_factor,
            dist,
        )

    def cal_potential_energy(self, batch: ProteinData):
        """
        Calculate the potential energy of the protein batch.
        :param batch: batch of `ProteinData`
        :return: a tensor of graph-level energy terms
        """
        dist = torch.norm(
            batch.pos[batch.nonbonded_atom_idx_src]
            - batch.pos[batch.nonbonded_atom_idx_dst],
            dim=-1,
        )
        energies = torch.stack(
            [
                self.get_bond_energy(batch),
                self.get_angle_energy(batch),
                self.get_dihedral_energy(batch),
                self.get_vdw_energy(batch, dist),
                self.get_elec_energy(batch, dist),
            ],
            dim=-1,
        )
        return energies

    def optimize_hydrogen(self, batch: ProteinData):
        """
        Optimize the given hydrogen atoms. Note that this function modifies
         the coordinates in-place.
        :param batch: batch of `ProteinData`
        :return: optimized batch
        """
        def closure():
            optimizer.zero_grad()
            positions = torch.cat([atom_pos, other_pos])
            batch.pos = positions[sort_idx]
            energy = self.cal_potential_energy(batch).sum()
            energy.backward()
            return energy

        device = batch.pos.device

        atom_pos = torch.nn.Parameter(batch.pos[batch.atom_idx])
        other_pos = batch.pos[batch.other_idx]
        all_idx = torch.cat([batch.atom_idx, batch.other_idx])
        sort_idx = torch.zeros_like(all_idx)
        sort_idx[all_idx] = torch.arange(len(all_idx), device=device)
        optimizer = torch.optim.LBFGS(
            [atom_pos],
            lr=0.1,
            max_iter=self.max_iter,
            tolerance_grad=0.1,
            tolerance_change=0.01,
        )
        optimizer.step(closure)
        batch.pos.detach_()
        return batch
