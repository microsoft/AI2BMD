import numpy as np
import torch
import torch_scatter


class DipeptideBondedCombiner:
    r"""
    Combine the energies and forces of the dipeptides and ACE-NMEs.
    """

    @staticmethod
    def energy_combine(
        dipeptides_energies: torch.Tensor,
        acenmes_energies: torch.Tensor,
    ) -> np.ndarray:
        r"""
        Combine the energies of the dipeptides and ACE-NMEs.
        """
        energy = torch.sum(dipeptides_energies) - torch.sum(acenmes_energies)

        return energy.detach().cpu().numpy()

    @staticmethod
    def forces_combine(
        num_atoms: int,
        dipeptides_forces: torch.Tensor,
        acenmes_forces: torch.Tensor,
        select_index: torch.Tensor,
        origin_index: torch.Tensor,
    ) -> np.ndarray:
        r"""
        Combine the forces of the dipeptides and ACE-NMEs, removing the extra
        forces that correspond to the added hydrogens, and assigning the
        selected forces to their original indices in the protein.
        """

        # negate ACE-NME forces
        forces = torch.cat([dipeptides_forces, -acenmes_forces])[select_index]
        forces = torch_scatter.scatter(forces, origin_index, dim=0, dim_size=num_atoms, reduce="sum")

        return forces.detach().cpu().numpy()


class DipeptideCombiner:
    @staticmethod
    def energy_combine(
        bonded_energy: np.float32, nonbonded_energy: np.float32
    ) -> np.float32:
        return bonded_energy + nonbonded_energy

    @staticmethod
    def forces_combine(
        bonded_forces: np.ndarray, nonbonded_forces: np.ndarray
    ) -> np.ndarray:
        return bonded_forces + nonbonded_forces
