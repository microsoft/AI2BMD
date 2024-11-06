import torch
import torch.nn.functional as F


class CTable:
    """
    The coefficient table as defined in the AMBER topology file `.prmtop`.
    The attributes share the same name as defined in the AMBER manual.
    See https://ambermd.org/FileFormats.php for more details.
    """

    def __init__(self):
        # Scalar values defined in `%FLAG POINTERS`
        self.natom = None
        self.ntypes = None
        self.numbnd = None
        self.numang = None
        self.nptra = None

        # 1D Arrays
        self.charge = None
        self.atomic_number = None
        self.atom_type_idx = None
        self.number_excluded_atoms = None
        self.nonbonded_parm_index = None
        self.bond_force_constant = None
        self.bond_equil_value = None
        self.angle_force_constant = None
        self.angle_equil_value = None
        self.dihedral_force_constant = None
        self.dihedral_periodicity = None
        self.dihedral_phase = None
        self.scee_scale_factor = None
        self.scnb_scale_factor = None
        self.lennard_jones_acoef = None
        self.lennard_jones_bcoef = None

        # 2D Arrays
        self.bonds_inc_hydrogen = None
        self.angles_inc_hydrogen = None
        self.dihedrals_inc_hydrogen = None
        self.excluded_atoms_list = None

    def print_stats(self):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                print(f"{name}: {value.size()}")
            else:
                print(f"{name}: {value}")

    def to(self, device):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    @classmethod
    def from_prmtop(cls, filename):
        """
        Read the coefficient table from the AMBER topology file `.prmtop`.
        :param filename: file name
        :return: a `CTable` instance
        """
        torch_dtype = {float: torch.float, int: torch.long}

        def _read_flag(flag, dtype=float, shape=None, as_tensor=True):
            """
            Read the values of a flag. A Flag is defined in a
            single line starting with `%FLAG`
            :param flag: flag string
            :param dtype: value type
            :param shape: value shape. If None is provided,
            the values are read as an 1D array.
            :param as_tensor: whether to return a tensor
            :return: a list or a tensor
            """
            nonlocal idx
            try:
                while not lines[idx].startswith(flag):
                    idx += 1
            except IndexError:
                raise ValueError(f"Cannot find flag {flag!r}")

            idx += 2
            values = []
            while idx < len(lines) and not lines[idx].startswith("%"):
                values.extend(map(dtype, lines[idx].split()))
                idx += 1
            if not as_tensor:
                return values
            values = torch.tensor(values, dtype=torch_dtype[dtype])
            if shape is not None:
                values = values.view(shape)
            return values

        ctable = cls()
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        idx = 0

        scalars = _read_flag("%FLAG POINTERS", int, as_tensor=False)
        ctable.natom = scalars[0]
        ctable.ntypes = scalars[1]
        ctable.numbnd = scalars[15]
        ctable.numang = scalars[16]
        ctable.nptra = scalars[17]

        # All indices are stored as 1-based in the prmtop file
        ctable.charge = _read_flag("%FLAG CHARGE", float)
        ctable.atomic_number = _read_flag("%FLAG ATOMIC_NUMBER", int)
        ctable.atom_type_idx = _read_flag("%FLAG ATOM_TYPE_INDEX", int) - 1
        ctable.number_excluded_atoms = _read_flag(
            "%FLAG NUMBER_EXCLUDED_ATOMS", int
        )
        ctable.nonbonded_parm_index = (
            _read_flag("%FLAG NONBONDED_PARM_INDEX", int) - 1
        )
        ctable.bond_force_constant = _read_flag(
            "%FLAG BOND_FORCE_CONSTANT", float
        )
        ctable.bond_equil_value = _read_flag("%FLAG BOND_EQUIL_VALUE", float)
        ctable.angle_force_constant = _read_flag(
            "%FLAG ANGLE_FORCE_CONSTANT", float
        )
        ctable.angle_equil_value = _read_flag("%FLAG ANGLE_EQUIL_VALUE", float)
        ctable.dihedral_force_constant = _read_flag(
            "%FLAG DIHEDRAL_FORCE_CONSTANT", float
        )
        ctable.dihedral_periodicity = _read_flag(
            "%FLAG DIHEDRAL_PERIODICITY", float
        )
        ctable.dihedral_phase = _read_flag("%FLAG DIHEDRAL_PHASE", float)
        ctable.scee_scale_factor = _read_flag("%FLAG SCEE_SCALE_FACTOR", float)
        ctable.scnb_scale_factor = _read_flag("%FLAG SCNB_SCALE_FACTOR", float)
        ctable.lennard_jones_acoef = _read_flag(
            "%FLAG LENNARD_JONES_ACOEF", float
        )
        ctable.lennard_jones_bcoef = _read_flag(
            "%FLAG LENNARD_JONES_BCOEF", float
        )

        # The bond, angle, and dihedral arrays store the atom indices as 3N
        ctable.bonds_inc_hydrogen = torch.div(
            _read_flag("%FLAG BONDS_INC_HYDROGEN", int, (-1, 3)),
            torch.LongTensor([[3, 3, 1]]),
            rounding_mode="floor",
        )
        ctable.angles_inc_hydrogen = torch.div(
            _read_flag("%FLAG ANGLES_INC_HYDROGEN", int, (-1, 4)),
            torch.LongTensor([[3, 3, 3, 1]]),
            rounding_mode="floor",
        )
        ctable.dihedrals_inc_hydrogen = torch.div(
            _read_flag("%FLAG DIHEDRALS_INC_HYDROGEN", int, (-1, 5)),
            torch.LongTensor([[3, 3, 3, 3, 1]]),
            rounding_mode="floor",
        )
        # Convert the atom indices to 0-based
        ctable.bonds_inc_hydrogen[:, -1] -= 1
        ctable.angles_inc_hydrogen[:, -1] -= 1
        ctable.dihedrals_inc_hydrogen[:, -1] -= 1
        # For unknown reason, there may be zeros in `excluded_atoms_list`
        ctable.excluded_atoms_list = (
            _read_flag("%FLAG EXCLUDED_ATOMS_LIST", int) - 1
        )
        return ctable

    def filter_bonds(self, atom_idx):
        """
        Filter the bonds that include the given hydrogen atoms.
        :param atom_idx: hydrogen atom indices to be considered
        :return: `bonds_inc_hydrogen` of shape (3, n_bonds)
        """
        mask = torch.isin(self.bonds_inc_hydrogen[:, :2], atom_idx).any(dim=-1)
        return self.bonds_inc_hydrogen[mask].t()

    def filter_angles(self, atom_idx):
        """
        Filter the angles that include the given hydrogen atoms.
        :param atom_idx: hydrogen atom indices to be considered
        :return: `angles_inc_hydrogen` of shape (4, n_angles)
        """
        mask = torch.isin(self.angles_inc_hydrogen[:, :3], atom_idx).any(
            dim=-1
        )
        return self.angles_inc_hydrogen[mask].t()

    def filter_dihedrals(self, atom_idx):
        """
        Filter the dihedrals that include the given hydrogen atoms.
        :param atom_idx: hydrogen atom indices to be considered
        :return: `dihedrals_inc_hydrogen` of shape (5, n_dihedrals)
        """
        mask = torch.isin(self.dihedrals_inc_hydrogen[:, :4], atom_idx).any(
            dim=-1
        )
        mask &= (self.dihedrals_inc_hydrogen[:, 2:4] >= 0).all(dim=-1)
        return self.dihedrals_inc_hydrogen[mask].t()

    def gen_nonbonded_pair(self, atom_idx):
        """
        Generate the non-bonded pair of the given hydrogen atoms.
        :param atom_idx: hydrogen atom indices to be considered
        :return: edge_index of shape (2, n_edges)
        """
        excluded_atoms_ptr = F.pad(
            torch.cumsum(self.number_excluded_atoms, dim=0),
            (1, 0),
            "constant",
            value=0,
        )

        all_edge_index = set(
            [
                (i, j)
                for i in range(self.natom)
                for j in range(i + 1, self.natom)
                if i in atom_idx or j in atom_idx
            ]
        )
        for atom_idx_src in range(self.natom):
            start, end = (
                excluded_atoms_ptr[atom_idx_src],
                excluded_atoms_ptr[atom_idx_src + 1],
            )
            exc_idx = self.excluded_atoms_list[start:end].tolist()
            for atom_idx_dst in exc_idx:
                # The excluded atom idx only contains atoms with larger indices
                all_edge_index.discard((atom_idx_src, atom_idx_dst))
        edge_index = torch.LongTensor(list(all_edge_index)).t()
        return edge_index

    def generate_lj_idx(self, atom_idx_src, atom_idx_dst):
        """
        Generate the Lennard-Jones index of the given hydrogen atoms.
        :param atom_idx_src: the source atom indices
        :param atom_idx_dst: the destination atom indices
        :return: indices of (n_edges,)
        """
        parm_idx = (
            self.ntypes * self.atom_type_idx[atom_idx_src]
            + self.atom_type_idx[atom_idx_dst]
        )
        return self.nonbonded_parm_index[parm_idx]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.natom} atoms,"
            f" {self.ntypes} types, "
            f"{self.numbnd} bonds, {self.numang} angles,"
            f" {self.nptra} dihedrals)"
        )
