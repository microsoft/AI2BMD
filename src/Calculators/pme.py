import helpmelib
import numpy as np
import torch
from ase.units import C, _eps0, kJ, mol, nm, pi
from torch.special import erf, erfc
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add

from AIMD.protein import Protein


def setup(
    pme, rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, numThreads
):
    """
    setup initializes this object for a PME calculation using only threading.
    This may be called repeatedly without compromising performance.
    :param pme: the PME instance.
    :param rPower: the exponent of the (inverse) distance kernel (e.g. 1 for Coulomb, 6 for attractive
            dispersion).
    :param kappa: the attenuation parameter in units inverse of those used to specify coordinates.
    :param splineOrder: the order of B-spline; must be at least (2 + max. multipole order + deriv. level needed).
    :param dimA: the dimension of the FFT grid along the A axis.
    :param dimB: the dimension of the FFT grid along the B axis.
    :param dimC: the dimension of the FFT grid along the C axis.
    :param scaleFactor: a scale factor to be applied to all computed energies and derivatives thereof (e.g. the
            1 / [4 pi epslion0] for Coulomb calculations).
    :param numThreads: the maximum number of threads to use for each MPI instance; if set to 0 all available threads
            are used.
    :return: None
    """
    pme.setup(
        rPower, kappa, splineOrder, dimA, dimB, dimC, scaleFactor, numThreads
    )


def set_lattice_vectors(
    pme, a, b, c, alpha, beta, gamma, latticeType=helpmelib.PMEInstanceF.LatticeType.XAligned
):
    """
    Sets the unit cell lattice vectors, with units consistent with those used to specify coordinates.
    :param pme: the PME instance.
    :param a: the A lattice parameter in units consistent with the coordinates.
    :param b: the B lattice parameter in units consistent with the coordinates.
    :param c: the C lattice parameter in units consistent with the coordinates.
    :param alpha: the alpha lattice parameter in degrees.
    :param beta: the beta lattice parameter in degrees.
    :param gamma: the gamma lattice parameter in degrees.
    :param latticeType: how to arrange the lattice vectors.  Options are
        ShapeMatrix: enforce a symmetric representation of the lattice vectors [c.f. S. Nosé and M. L. Klein,
                    Mol. Phys. 50 1055 (1983)] particularly appendix C.
        XAligned: make the A vector coincide with the X axis, the B vector fall in the XY plane, and the C vector
                 take the appropriate alignment to completely define the system.
    :return: None
    """
    pme.set_lattice_vectors(a, b, c, alpha, beta, gamma, latticeType)


def compute_E_rec(pme, mat, parameterAngMom, parameters, coordinates):
    """
    Runs a PME reciprocal space calculation, computing energies and forces.
    :param pme: the PME instance.
    :param mat: the matrix type (either MatrixD or MatrixF).
    :param parameterAngMom: the angular momentum of the parameters (0 for charges, C6 coefficients, 2 for quadrupoles, etc.).
    :param parameters: parameters the list of parameters associated with each atom (charges, C6
        coefficients, multipoles, etc...). For a parameter with angular momentum L, a matrix of dimension nAtoms x nL
        is expected, where nL = (L+1)*(L+2)*(L+3)/6 and the fast running index nL has the ordering
        0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ ...
    :param coordinates: the cartesian coordinates, ordered in memory as {x1,y1,z1,x2,y2,z2,....xN,yN,zN}.
    :return: the reciprocal space energy.
    """
    return pme.compute_E_rec(
        parameterAngMom, mat(parameters), mat(coordinates)
    )


def electro_dir(coord, charge, edge_index, beta=1.0):
    if edge_index[0].size(0) == 0:
        return 0
    src, dst = edge_index
    dist = torch.norm(coord[dst] - coord[src], dim=1)
    discount = erfc(beta * dist)
    return 0.5 * (charge[src] * charge[dst] * discount / dist).sum()


def electro_slf(charge, beta=1.0):
    return -beta / np.sqrt(np.pi) * (charge**2).sum()


def electro_adj(coord, charge, exclude_index, beta=1.0):
    if exclude_index[0].size(0) == 0:
        return 0
    src, dst = exclude_index
    dist = torch.norm(coord[dst] - coord[src], dim=1)
    discount = erf(beta * dist)
    return -0.5 * (charge[src] * charge[dst] * discount / dist).sum()


def electro_neutral(pme, charge, beta):
    # return -np.pi / 2 / beta ** 2 / volume * (charge.sum() ** 2)
    q_tot = charge.sum()
    rec_sp = compute_E_rec(
        pme,
        helpmelib.MatrixF,
        0,
        q_tot.reshape(1, 1),
        np.zeros((1, 3), dtype=np.float32),
    )
    self_sp = -beta * q_tot**2 / np.sqrt(np.pi)
    return -rec_sp - self_sp


class PMENonBondedCalculator:
    def __init__(self, beta=0.3, cutoff=9.0, grid_spacing=1.0, device="cpu") -> None:
        super().__init__()
        self.beta = beta
        self.cutoff = cutoff
        self.grid_spacing = grid_spacing
        self.device = device
        self.k = 1 / (4 * pi * _eps0) * 10e6 * mol * C ** (-2)
        self.pme = helpmelib.PMEInstanceF()

        self.sigmas = None
        self.epsilons = None
        self.charges = None
        self.charges_cpu = None
        self.src_ex = None
        self.dst_ex = None
        self.coulomb_slf = 0.0
        self.coulomb_neutral = 0.0

    def set_parameters(self, prot: Protein) -> None:
        # Initialize parameters
        sigmas = torch.tensor(prot.sigmas, dtype=torch.float)
        epsilons = torch.tensor(prot.epsilons, dtype=torch.float)
        charges = torch.tensor(prot.charges, dtype=torch.float)
        self.sigmas = sigmas.to(self.device)
        self.epsilons = epsilons.to(self.device)
        self.charges = charges.to(self.device)
        self.charges_cpu = prot.charges[:, None]
        src_ex, dst_ex = prot.exclude_index
        self.src_ex = src_ex.to(self.device)
        self.dst_ex = dst_ex.to(self.device)

        # Initialize PME
        cx, cy, cz = prot.cell.diagonal()
        dimA = int(cx / self.grid_spacing)
        dimB = int(cy / self.grid_spacing)
        dimC = int(cz / self.grid_spacing)
        setup(self.pme, 1, self.beta, 4, dimA, dimB, dimC, 1, 0)
        set_lattice_vectors(self.pme, cx, cy, cz, 90, 90, 90)
        self.coulomb_slf = electro_slf(prot.charges, self.beta)
        self.coulomb_neutral = electro_neutral(self.pme, prot.charges, self.beta)

    def __call__(self, prot: Protein) -> tuple[np.float32, np.ndarray]:
        r"""
        Using non-bonded atom pairs to calculate
        the non-bonded energy and force.
        Non-bonded forces are calculated by grading the non-bonded energy
        with respect of the atom positions.
        """
        pos_cpu = torch.tensor(prot.get_positions(), dtype=torch.float)
        pos = pos_cpu.to(self.device)
        src, dst = radius_graph(
            pos, self.cutoff, max_num_neighbors=len(pos), loop=False
        )
        vec = pos[dst] - pos[src]
        d2 = (vec**2).sum(-1)
        d = torch.sqrt(d2)
        vec_ex = pos[self.dst_ex] - pos[self.src_ex]
        d2_ex = (vec_ex**2).sum(-1)
        d_ex = torch.sqrt(d2_ex)

        # LJ
        sigmaij = 0.5 * (self.sigmas[src] + self.sigmas[dst]) * nm
        epsij = torch.sqrt(self.epsilons[src] * self.epsilons[dst])
        c6 = (sigmaij**2 / d2) ** 3
        c12 = c6**2
        energy_lj = 4 * epsij * (c12 - c6) / 2
        force_lj = (24 * epsij * (2 * c12 - c6) / d2).unsqueeze(-1) * vec

        # LJ exclude
        sigmaij_ex = 0.5 * (self.sigmas[self.src_ex] + self.sigmas[self.dst_ex]) * nm
        epsij_ex = torch.sqrt(self.epsilons[self.src_ex] * self.epsilons[self.dst_ex])
        c6_ex = (sigmaij_ex**2 / d2_ex) ** 3
        c12_ex = c6_ex**2
        energy_lj_ex = 4 * epsij_ex * (c12_ex - c6_ex) / 2
        force_lj_ex = (24 * epsij_ex * (2 * c12_ex - c6_ex) / d2_ex).unsqueeze(-1) * vec_ex

        # Coulomb
        energy_coulomb_rec = compute_E_rec(self.pme, helpmelib.MatrixF, 0, self.charges_cpu, pos_cpu)
        energy_coulomb_dir = electro_dir(pos, self.charges, (src, dst), self.beta)
        energy_coulomb_adj = electro_adj(pos, self.charges, prot.exclude_index, self.beta)
        energy_coulomb_ex = electro_dir(pos, self.charges, (self.src_ex, self.dst_ex), self.beta)
        energy_coulomb = (
            energy_coulomb_rec
            + energy_coulomb_dir
            + energy_coulomb_adj
            + self.coulomb_slf
            + self.coulomb_neutral
            - energy_coulomb_ex
        ) * self.k
        force_coulomb = (self.k * self.charges[src] * self.charges[dst] / d / d2).unsqueeze(-1) * vec
        force_coulomb_ex = (
            self.k * self.charges[self.src_ex] * self.charges[self.dst_ex] / d_ex / d2_ex
        ).unsqueeze(-1) * vec_ex

        # Combine
        energy = energy_lj.sum() - energy_lj_ex.sum() + energy_coulomb
        force = scatter_add(force_lj + force_coulomb, dst, dim=0, dim_size=len(prot))
        force_ex = scatter_add(force_lj_ex + force_coulomb_ex, self.dst_ex, dim=0, dim_size=len(prot))
        energy = energy.item() * (kJ / mol)
        force = (force - force_ex).cpu().numpy() * (kJ / mol)
        return energy, force
