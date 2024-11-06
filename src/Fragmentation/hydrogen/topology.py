from typing import Optional

import torch
from torch_geometric.data import Batch, Data, separate

from .ctable import CTable


class ProteinData(Data):
    """
    Class representing a protein structure as a graph.
    This class modifies the `__inc__` method so that
    the graph attributes can be correctly batched.
    Note that `pyg` requires a data class to have a
    default constructor to be batchable.
    Therefore, all parameters passed to the constructor must be optional.
    """

    def __init__(
        self, atom_idx=None, other_idx=None, pos=None, ctable: Optional[CTable] = None
    ):
        """
        Initialize a `ProteinData` object representing a protein graph.
        :param atom_idx: the hydrogen atom indices to be considered
        :param pos: the coordinates of the whole protein
        :param ctable: a `CTable` object containing the protein information
        """

        def attr(name):
            """Get an attribute from CTable or return None if ctable is None."""
            return None if ctable is None else getattr(ctable, name)

        def method(name, return_cnt):
            """Get a method call result from CTable or return a
            list of None's if ctable is None."""
            return (
                [None] * return_cnt
                if ctable is None
                else getattr(ctable, name)(atom_idx)
            )

        def zeros_like(name):
            """Get a tensor of zeros or return None if ctable is None."""
            return (
                None
                if ctable is None
                else torch.zeros_like(getattr(self, name))
            )

        super().__init__(pos=pos, z=attr("atomic_number"))
        self.atom_idx = atom_idx
        self.other_idx = other_idx

        # The following scalar attributes are only used
        # for batching data. Make them private.
        self._natom = attr("natom")
        self._ntypes = attr("ntypes")
        self._numbnd = attr("numbnd")
        self._numang = attr("numang")
        self._nptra = attr("nptra")

        # The following attributes can be just concatenated.
        self.charge = attr("charge")
        self.bond_force_constant = attr("bond_force_constant")
        self.bond_equil_value = attr("bond_equil_value")
        self.angle_force_constant = attr("angle_force_constant")
        self.angle_equil_value = attr("angle_equil_value")
        self.dihedral_force_constant = attr("dihedral_force_constant")
        self.dihedral_periodicity = attr("dihedral_periodicity")
        self.dihedral_phase = attr("dihedral_phase")
        self.lennard_jones_acoef = attr("lennard_jones_acoef")
        self.lennard_jones_bcoef = attr("lennard_jones_bcoef")

        # The following indices need to be filtered by atom_idx.
        # They need to be increased during batching.
        (
            self.bonds_atom_idx_src,
            self.bonds_atom_idx_dst,
            self.bond_idx,
        ) = method("filter_bonds", 3)
        (
            self.angles_atom_idx_i,
            self.angles_atom_idx_j,
            self.angles_atom_idx_k,
            self.angle_idx,
        ) = method("filter_angles", 4)
        (
            self.dihedrals_atom_idx_i,
            self.dihedrals_atom_idx_j,
            self.dihedrals_atom_idx_k,
            self.dihedrals_atom_idx_l,
            self.dihedral_idx,
        ) = method("filter_dihedrals", 5)
        self.nonbonded_atom_idx_src, self.nonbonded_atom_idx_dst = method(
            "gen_nonbonded_pair", 2
        )
        self.lj_idx = (
            None
            if ctable is None
            else ctable.generate_lj_idx(
                self.nonbonded_atom_idx_src, self.nonbonded_atom_idx_dst
            )
        )

        # The following attributes contains the batch indices.
        # They need to be increased by 1 during batching.
        self.bond_batch = zeros_like("bond_idx")
        self.angle_batch = zeros_like("angle_idx")
        self.dihedral_batch = zeros_like("dihedral_idx")
        self.nonbonded_batch = zeros_like("lj_idx")

    def __inc__(self, key, value, *args, **kwargs):
        if "atom_idx" in key:
            # All atom indices need to be increased by the number
            # of nodes during batching.
            return self._natom
        elif "other_idx" in key:
            return self._natom
        elif "_batch" in key:
            # All batch indices need to be increased by 1 during batching.
            return 1
        elif key == "bond_idx":
            return self._numbnd
        elif key == "angle_idx":
            return self._numang
        elif key == "dihedral_idx":
            return self._nptra
        elif key == "lj_idx":
            return self._ntypes * (self._ntypes + 1) // 2
        return super().__inc__(key, value, *args, **kwargs)


class ProteinDataBatch(Batch, ProteinData):
    """
    Class representing a batch of protein structures as a graph.
    This class modifies the `get_example` method so that
    only the specified attributes are unbatched to save memory.
    """

    def get_example(self, idx, keys=("pos", "z")):
        """
        Get the `idx`-th example from the batch. Only the specified
         attributes are unbatched.
        :param idx: data index
        :param keys: the attributes to be unbatched. If None, all
        attributes are unbatched.
        :return: a `ProteinData` object
        """
        if not hasattr(self, "_slice_dict"):
            raise RuntimeError(
                (
                    "Cannot reconstruct 'Data' object from 'Batch' because "
                    "'Batch' was not created via 'Batch.from_data_list()'"
                )
            )

        slice_dict = (
            self._slice_dict
            if keys is None
            else {key: self._slice_dict[key] for key in keys}
        )
        data = separate.separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        return data

    def to_data_list(self, keys=("pos", "z")):
        """
        Convert a batch to a list of `ProteinData` objects.
        :param keys: the attributes to be unbatched. If None,
        all attributes are unbatched.
        :return: a list of `ProteinData` objects
        """
        return [self.get_example(i, keys) for i in range(self.num_graphs)]
