import numpy as np
import torch
from ase.units import C, _eps0, kJ, mol, nm, pi
from torch_scatter import scatter_add

from AIMD.protein import Protein


class MMNonBondedCalculator:
    r"""
    MMNonBondedCalculator is a non-bonded calculator based on MM calculations.
    """

    def __init__(self, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.k = 1 / (4 * pi * _eps0) * 10e6 * mol * C ** (-2)
        self.sigmas = None
        self.epsilons = None
        self.charges = None
        self.src = None
        self.dst = None

    def set_parameters(self, prot: Protein) -> None:
        self.sigmas = torch.tensor(prot.sigmas, dtype=torch.float, device=self.device)
        self.epsilons = torch.tensor(prot.epsilons, dtype=torch.float, device=self.device)
        self.charges = torch.tensor(prot.charges, dtype=torch.float, device=self.device)

        src, dst = prot.initial_mm_adjmatrix()
        self.src = src.to(self.device)
        self.dst = dst.to(self.device)

    def __call__(self, prot: Protein) -> tuple[np.float32, np.ndarray]:
        r"""
        Using non-bonded atom pairs to calculate the non-bonded energy and
        force. Non-bonded forces are calculated by calculating the gradient of
        the non-bonded energy with respect to the atom positions.
        """
        pos = torch.tensor(prot.get_positions(), dtype=torch.float, device=self.device)

        vec = pos[self.dst] - pos[self.src]
        d2 = (vec**2).sum(-1)
        d = torch.sqrt(d2)

        # LJ
        sigmaij = 0.5 * (self.sigmas[self.src] + self.sigmas[self.dst]) * nm
        epsij = torch.sqrt(self.epsilons[self.src] * self.epsilons[self.dst])
        c6 = (sigmaij**2 / d2) ** 3
        c12 = c6**2
        energy_lj = 4 * epsij * (c12 - c6)
        force_lj = (24 * epsij * (2 * c12 - c6) / d2).unsqueeze(-1) * vec

        # Coulomb
        energy_coulomb = self.k * self.charges[self.src] * self.charges[self.dst] / d
        force_coulomb = (energy_coulomb / d2).unsqueeze(-1) * vec

        # Combine
        energy = energy_lj.sum() + energy_coulomb.sum()
        force = force_lj + force_coulomb
        force = scatter_add(force, self.dst, dim=0, dim_size=len(prot))
        energy = energy.cpu().item() * (kJ / mol) / 2
        force = force.cpu().numpy() * (kJ / mol)
        return energy, force
