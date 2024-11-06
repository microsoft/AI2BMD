import copy
import numbers
import os
from itertools import product
from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.atom import Atom
from openmm import NonbondedForce
from openmm.app import ForceField, NoCutoff, PDBFile


class Protein(Atoms):
    def __init__(
        self,
        atoms: Atoms,
        pdb4params: Optional[str] = None,
        charges: Optional[np.ndarray] = None,
        sigmas: Optional[np.ndarray] = None,
        epsilons: Optional[np.ndarray] = None,
    ):
        # * Initialize the Atoms properties
        self.__dict__.update(atoms.__dict__)

        assert pdb4params is not None, "pdb4params is not specified."

        self.nowater_PDB_path = pdb4params
        self.charges = charges
        self.sigmas = sigmas
        self.epsilons = epsilons
        # skip_check_state: if True, skips the atom comparison and calculation in Calculator.get_property(...)
        # can be overridden with utils.SkipCheckState context manager.
        self.skip_check_state = False

        if self.charges is None or self.sigmas is None or self.epsilons is None:
            self.generate_nonbonded_params()

    def __getitem__(self, i):
        r"""Return a subset of the atoms.

        i -- scalar integer, list of integers, or slice object
        describing which atoms to return.

        If i is a scalar, return an Atom object. If i is a list or a
        slice, return an Atoms object with the same cell, pbc, and
        other associated info as the original Atoms object. The
        indices of the constraints will be shuffled so that they match
        the indexing in the subset returned.

        """

        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError("Index out of range.")

            return Atom(atoms=self, index=i)
        elif not isinstance(i, slice):
            i = np.array(i)
            # if i is a mask
            if i.dtype == bool:
                if len(i) != len(self):
                    raise IndexError(
                        "Length of mask {} must equal "
                        "number of atoms {}".format(len(i), len(self))
                    )
                i = np.arange(len(self))[i]

        import copy

        conadd = []
        # Constraints need to be deepcopied, but only the relevant ones.
        for con in copy.deepcopy(self.constraints):
            try:
                con.index_shuffle(self, i)
            except (IndexError, NotImplementedError):
                pass
            else:
                conadd.append(con)

        atoms = Atoms(
            cell=self.cell,
            pbc=self.pbc,
            info=self.info,
            celldisp=self._celldisp.copy(),
        )
        # TODO: Do we need to shuffle indices in adsorbate_info too?

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a[i].copy()

        atoms.constraints = conadd
        copy_object = self.__class__(
            atoms,
            pdb4params=self.nowater_PDB_path,
            charges=self.charges,
            sigmas=self.sigmas,
            epsilons=self.epsilons,
        )
        return copy_object

    def copy(self):
        r"""
        Return a copy.
        """
        atoms = Atoms(
            cell=self.cell,
            pbc=self.pbc,
            info=self.info,
            celldisp=self._celldisp.copy(),
        )

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        atoms.constraints = copy.deepcopy(self.constraints)
        copy_object = self.__class__(
            atoms,
            pdb4params=self.nowater_PDB_path,
            charges=self.charges,
            sigmas=self.sigmas,
            epsilons=self.epsilons,
        )
        return copy_object

    @property
    def num_atoms(self):
        return len(self)

    def initial_mm_adjmatrix(self):
        r"""
        Mask the previous atoms or the atoms in the same dipeptides,
         and return the adjacency matrix.
        """

        # Get the total number of nodes (atoms)
        num_nodes = len(self.positions)

        # Generate all possible pairs of nodes
        pairs = [
            (i, j) for i, j in product(range(num_nodes), repeat=2) if i != j
        ]
        # Remove the pairs that are in the same dipeptide
        edge_index = torch.tensor(
            [p for p in pairs if p not in self.exclude_pair],
            dtype=torch.long,
        ).t()
        return edge_index

    def generate_nonbonded_params(self):
        r"""
        Use OpenMM for generating non-bonded parameters.
        """
        pdb = PDBFile(self.nowater_PDB_path)
        forcefield = ForceField("amber14-all.xml")
        system = forcefield.createSystem(
            pdb.topology, nonbondedMethod=NoCutoff
        )
        nonbonded = [
            f for f in system.getForces() if isinstance(f, NonbondedForce)
        ][0]
        charge_ls = []
        sigma_ls = []
        epsilon_ls = []
        for i in range(system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charge_ls.append(charge._value)
            sigma_ls.append(sigma._value)
            epsilon_ls.append(epsilon._value)
        self.charges = np.array(charge_ls, dtype=np.float32)
        self.sigmas = np.array(sigma_ls, dtype=np.float32)
        self.epsilons = np.array(epsilon_ls, dtype=np.float32)
