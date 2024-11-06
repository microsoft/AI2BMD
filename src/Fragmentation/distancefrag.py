import gzip
import os
import pickle
from itertools import chain, combinations
from typing import Any

import numpy as np
import torch
from ase import Atoms

from AIMD import arguments
from AIMD.fragment import FragmentData
from AIMD.preprocess import Preprocess
from AIMD.protein import Protein
from Calculators.device_strategy import DeviceStrategy
from Fragmentation.basefrag import DipeptideFragment
from Fragmentation.hydrogen import CTable, HydrogenOptimizer, ProteinData, ProteinDataBatch
from utils.reference import fragment_atomic_numbers, fragment_info
from utils.utils import numpy_to_torch, numpy_list_to_torch


class DistanceFragment(DipeptideFragment):
    r"""
    DistanceFragment is a subclass of DipeptideFragment.
    It fragments a protein into dipeptides and ACE-NMEs
    according to atom radius and previous atom position.

    """

    def __init__(self, max_iter=10) -> None:
        super().__init__()
        self.optimizer = HydrogenOptimizer(max_iter)

    @staticmethod
    def get_dipeptide_positions(prot: Protein, device: str) -> torch.tensor:
        r"""
        Obtain dipeptide positions based on the list of added hydrogen and
        hydrogen acceptor indices.
        """

        positions = torch.tensor(prot.arrays["positions"], dtype=torch.float, device=device)

        dipeptide_positions = positions[prot.all_dipeptide_index]
        hydrogen_positions = positions[prot.all_hydrogen_index]
        acceptor_positions = positions[prot.all_acceptor_index]
        direction = hydrogen_positions - acceptor_positions
        direction = direction / torch.norm(direction, dim=1, keepdim=True)
        hydrogen_positions = acceptor_positions + direction * prot.all_hydrogen_radii

        positions = torch.zeros((prot.dipeptides_len, 3), dtype=torch.float, device=device)
        positions = positions.scatter(0, prot.scatter_original_index, dipeptide_positions)
        positions = positions.scatter(0, prot.scatter_hydrogen_index, hydrogen_positions)

        return positions

    def get_fragments(self, prot: Protein) -> FragmentData:
        r"""
        Get dipeptide and ACE-NME fragments, given the current atom positions.
        The coordinates of the added hydrogen atoms are optimised on the
        dipeptides, before extracting the ACE-NME coordinats.

        Parameters:
        -----------
            prot: Protein
                The protein to be fragmented.

        Returns:
        --------
            fragments: FragmentData
                The combined fragmented dipeptides and ACE-NMEs of the protein.
        """

        # retrieve device for hydrogen optimiser
        device = DeviceStrategy.get_optimiser_device()

        self.batch.pos = self.get_dipeptide_positions(prot, device)

        # * minimize the dipeptides
        self.optimizer.optimize_hydrogen(self.batch)

        positions = self.batch.pos.cpu().numpy()
        positions = positions[prot.fragments_index]

        fragments = FragmentData(
            prot.fragments_z,
            positions,
            prot.fragments_start,
            prot.fragments_end,
            prot.fragments_batch,
        )

        return fragments

    def fragment(self, prot: Protein) -> None:
        r"""
        Fragment a protein into dipeptides and ACE-NMEs according to atom
        radius and previous atom position.

        Parameters:
        -----------
            prot: Atoms
                The protein to be fragmented.
        """

        # a. fragment the protein into dipeptides and ACE-NMEs
        dipeptides_index, acenmes_index = self.get_fragments_index(prot)

        dipeptides_count = len(dipeptides_index)
        acenmes_count = len(acenmes_index)

        # b. calculate locations where hydrogens should be added
        prot.atom_masks = {
            "N": prot.arrays["atomtypes"] == "N",
            "C": prot.arrays["atomtypes"] == "C",
            "CA": prot.arrays["atomtypes"] == "CA",
            "CB": prot.arrays["atomtypes"] == "CB",
            "CD": prot.arrays["atomtypes"] == "CD",
        }

        hydrogens_radii, acceptors_index, hydrogens_index = map(
            list,
            zip(
                *[
                    self.get_hydrogen_indices(prot, i, dipeptide_idx)
                    for i, dipeptide_idx in enumerate(dipeptides_index)
                ]
            )
        )

        # c. calculate lengths of dipeptides (original atoms + hydrogens), ACE-NMEs
        dipeptides_length = [len(d) + len(h) for d, h in zip(dipeptides_index, hydrogens_index)]
        acenmes_length = [12] * acenmes_count

        # store info of original (separate CYX) dipeptides so that
        #     1. added hydrogens are placed at the correct indices
        #     2. relative indices of ace-nmes are correct
        hydrogens_split = dipeptides_length.copy()
        hydrogens_offset = [len(d) for d in dipeptides_index]

        # d. build residue info list
        residuenames = [name.strip() for name in prot.arrays["residuenames"]]

        prot.resi_info = [
            [
                residuenames[dipeptides_index[0][6]],
                0,
                residuenames[dipeptides_index[1][6]],
                residuenames[dipeptides_index[0][0]],
            ]
        ]
        for i in range(1, dipeptides_count - 1):
            prot.resi_info.append(
                [
                    residuenames[dipeptides_index[i][6]],
                    2,
                    residuenames[dipeptides_index[i + 1][6]],
                    residuenames[dipeptides_index[i][0]],
                ]
            )
        prot.resi_info.append(
            [
                residuenames[dipeptides_index[-1][6]],
                1,
                residuenames[dipeptides_index[-1][-1]],
                residuenames[dipeptides_index[-2][6]],
            ]
        )

        # e. calculate quantities related to hydrogen optimisation
        #     1. permute_index: permute atoms to correct index
        #     2. constrain_index: indices of hydrogens after permutation
        permute_index, constrain_index = map(
            list,
            zip(
                *[
                    self.calculate_permutation_indices(prot, i, length)
                    for i, length in enumerate(dipeptides_length)
                ]
            )
        )

        # f. calculate indices of original atoms in dipeptides
        select_index = self.calculate_select_indices(permute_index, constrain_index)

        # z. adjust indices to handle CYX residues
        #     CYX-CYX residues are treated as a single residue, with the second
        #     (by residue number) residue zeroed out

        S_bonds = self.get_cystine_bonds(prot, dipeptides_index)

        if len(S_bonds):
            if arguments.get().verbose >= 1:
                print(f" [i] S-S bonds: {S_bonds}")

            # combine CYX residue pairs, leave second dipeptide empty
            for i, j in S_bonds.items():
                offset = dipeptides_length[i]

                # adjust dipeptides_index
                dipeptides_index[i].extend(dipeptides_index[j])
                dipeptides_index[j] = []

                # adjust dipeptides_length
                dipeptides_length[i] = dipeptides_length[i] + dipeptides_length[j]
                dipeptides_length[j] = 0

                # adjust select_index
                select_index[i] = np.concatenate([select_index[i], select_index[j] + offset])
                select_index[j] = np.array([], dtype=int)

                # adjust permute_index
                permute_index[i] = np.concatenate([permute_index[i], permute_index[j] + offset])
                permute_index[j] = np.array([], dtype=int)

                # adjust constrain_index
                constrain_index[i].extend([x + offset for x in constrain_index[j]])
                constrain_index[j] = []

                # adjust residue names
                prot.resi_info[j][0] = "CYZ"

            shuffle_index = []
            for i in range(dipeptides_count):
                if i not in S_bonds.values():
                    shuffle_index.append(i)
                if i in S_bonds:
                    shuffle_index.append(S_bonds[i])

            def _adjust_CYX_pair_order(data: list[Any], index: list[int]):
                return [data[index[i]] for i in range(len(data))]

            # adjust hydrogen info
            hydrogens_split = _adjust_CYX_pair_order(hydrogens_split, shuffle_index)
            hydrogens_offset = _adjust_CYX_pair_order(hydrogens_offset, shuffle_index)

            hydrogens_radii = _adjust_CYX_pair_order(hydrogens_radii, shuffle_index)
            acceptors_index = _adjust_CYX_pair_order(acceptors_index, shuffle_index)
            hydrogens_index = _adjust_CYX_pair_order(hydrogens_index, shuffle_index)
        else:
            shuffle_index = list(range(dipeptides_count))

        # g. build protein graph for hydrogen optimisation
        protein_data = self.create_protein_graph(prot.resi_info, dipeptides_length, constrain_index)

        device = DeviceStrategy.get_optimiser_device()

        self.batch = ProteinDataBatch.from_data_list(protein_data).to(device)

        # h. calculate concatenated, interleaved fragments info
        def interleave(dipeptides: list[Any], acenmes: list[Any]):
            output = [None] * (len(dipeptides) + len(acenmes))
            output[0::2] = dipeptides
            output[1::2] = acenmes

            return output

        dipeptides_len = np.sum(dipeptides_length)

        dipeptides_z = [fragment_atomic_numbers[info[0]] for info in prot.resi_info]
        dipeptides_start = np.zeros((dipeptides_count,), dtype=int)
        dipeptides_end = np.cumsum(dipeptides_length)
        dipeptides_start[1:] = dipeptides_end[:-1]

        acenme_atomic_numbers = np.array([1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1])

        acenmes_z = np.tile(acenme_atomic_numbers, (acenmes_count, 1))

        fragments_length = interleave(dipeptides_length, acenmes_length)
        fragments_count = dipeptides_count + acenmes_count

        fragments_z = interleave(dipeptides_z, acenmes_z)
        fragments_start = np.zeros((fragments_count,), dtype=int)
        fragments_end = np.cumsum(fragments_length)
        fragments_start[1:] = fragments_end[:-1]
        fragments_batch = np.zeros((fragments_end[-1],), dtype=int)
        fragments_batch[fragments_start[1:]] = 1
        fragments_batch = np.cumsum(fragments_batch)

        prot.dipeptides_len = dipeptides_len

        prot.fragments_z = np.concatenate(fragments_z)
        prot.fragments_start = fragments_start
        prot.fragments_end = fragments_end
        prot.fragments_batch = fragments_batch

        # i. calculate relative indices for acenmes
        hydrogens_start = np.zeros((dipeptides_count,), dtype=int)
        hydrogens_end = np.cumsum(hydrogens_split)
        hydrogens_start[1:] = hydrogens_end[:-1]

        dipeptide_acenme_indices = [
            [
                np.arange(start, start + 6, dtype=int),
                np.arange(end - 6, end, dtype=int),
            ]
            for start, end in zip(hydrogens_start, hydrogens_end)
        ]

        inverse_shuffle_index = np.argsort(shuffle_index)
        acenmes_relative_index = np.flip(np.concatenate(list(chain(*[
            dipeptide_acenme_indices[inverse_shuffle_index[i]] for i in range(dipeptides_count)
        ]))[1:-1]).reshape(-1, 2, 6), axis=1).reshape(-1, 12).tolist()

        prot.fragments_index = np.concatenate(interleave(
            np.split(np.arange(dipeptides_len, dtype=int), dipeptides_end[:-1]),
            acenmes_relative_index,
        ))

        # j. calculate concatenated indices for calculating dipeptide positions
        scatter_original_index = []
        scatter_hydrogen_index = []
        gather_dipeptide_index = []

        for start, permute_idx in zip(dipeptides_start, permute_index):
            gather_dipeptide_index.append(start + torch.tensor(permute_idx))

        for start, offset, hydrogen_idx in zip(hydrogens_start, hydrogens_offset, hydrogens_index):
            scatter_original_index.append(start + torch.arange(offset))
            scatter_hydrogen_index.append(start + offset + torch.arange(len(hydrogen_idx)))

        argsort_index = torch.argsort(torch.cat(gather_dipeptide_index))
        scatter_original_index = argsort_index[torch.cat(scatter_original_index)].reshape(-1, 1)
        scatter_hydrogen_index = argsort_index[torch.cat(scatter_hydrogen_index)].reshape(-1, 1)

        dipeptides_index = [np.array(idx, dtype=int) for idx in dipeptides_index]

        prot.all_dipeptide_index = numpy_list_to_torch(dipeptides_index, device)
        prot.all_hydrogen_radii = numpy_list_to_torch(hydrogens_radii, device)
        prot.all_hydrogen_index = numpy_list_to_torch(hydrogens_index, device)
        prot.all_acceptor_index = numpy_list_to_torch(acceptors_index, device)

        prot.scatter_original_index = scatter_original_index.expand(-1, 3).to(device)
        prot.scatter_hydrogen_index = scatter_hydrogen_index.expand(-1, 3).to(device)

        # k. calculate concatenated indices for force combination
        device = DeviceStrategy.get_default_device()

        origin_index = dipeptides_index + acenmes_index

        sizes = numpy_to_torch(np.cumsum([len(v) for v in select_index]), device)
        length = dipeptides_length + acenmes_length
        length = torch.tensor(length, dtype=torch.int, device=device)

        offset = torch.zeros((sizes[-1],), dtype=torch.int, device=device)
        offset = torch.cumsum(offset.scatter(0, sizes[:-1], length), dim=0)

        select_index = numpy_list_to_torch(select_index, device)
        origin_index = numpy_list_to_torch(origin_index, device)

        select_index = select_index + offset

        prot.select_index = select_index
        prot.origin_index = origin_index

        # l. calculate pair indices for atoms within the same dipeptide
        exclude_pair = set()
        for dipeptide_index in dipeptides_index:
            for x, y in combinations(dipeptide_index, 2):
                exclude_pair.add((x, y))
                exclude_pair.add((y, x))

        prot.exclude_pair = exclude_pair
        prot.exclude_index = torch.tensor(list(prot.exclude_pair), dtype=torch.long).t()

    @staticmethod
    def get_hydrogen_indices(
        prot: Atoms, idx: int, dipeptide_idx: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Find indices at which to add hydrogens to the fragmented dipeptide.

        Parameters:
        -----------
            prot: Atoms
                The protein to be fragmented.
            idx: int
                The index of the fragmented dipeptide.
            dipeptide_idx: list[int]
                The indices of atoms in the original protein that belong to
                this dipeptide.

        """
        element2radius = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
        }  # atom radius for the addition during distance based fragment

        radii = []
        acceptor_idx = []
        hydrogen_idx = []

        def residue_match(atom, fidx_mask, err_res_idx, err_idx_off):
            match = prot.atom_masks[atom] & fidx_mask
            if np.any(match):
                return np.argmax(match)

            res = prot.arrays["residuenames"][dipeptide_idx[err_res_idx]].strip()
            raise ValueError(
                f"Error: No atom name {atom} found in the residue "
                f"{res} {idx + err_idx_off}"
            )

        fidx_match_p_1 = prot.arrays["residuenumbers"] == idx + 1
        fidx_match_p_3 = prot.arrays["residuenumbers"] == idx + 3

        # * The following operations add hydrogens on the N-terminal of
        # dipeptide, proline has no specific issue here
        if prot.arrays["residuenames"][dipeptide_idx[0]].strip() == "GLY":
            # * the first residue in the dipeptide is a GLY (glycine),
            # so only 1 hydrogen should be added on it,
            # (CA-N) is replaced by CA-H
            acceptor_posi = residue_match("CA", fidx_match_p_1, 0, 1)
            previous_atom_posi = residue_match("N", fidx_match_p_1, 0, 1)

            # * Set the indices of the additional hydrogen
            radii.append(element2radius["C"] + element2radius["H"])
            acceptor_idx.append(acceptor_posi)
            hydrogen_idx.append(previous_atom_posi)

        elif prot.arrays["residuenames"][dipeptide_idx[0]].strip() != "ACE":
            # * the first residue in the dipeptide is NOT a GLY and not an ACE
            # (ACE should not be added for hydrogen), so 2 hydrogens should be
            # added on it, (CA-N) is replaced by CA-H,
            # (CA-CB) is replaced by CA-H
            acceptor_posi = residue_match("CA", fidx_match_p_1, 0, 1)
            previous_atom_posi_N = residue_match("N", fidx_match_p_1, 0, 1)
            previous_atom_posi_CB = residue_match("CB", fidx_match_p_1, 0, 1)

            # * Set the indices of the additional hydrogen (CA-N)
            # * Set the indices of the additional hydrogen (CA-CB)
            radii.append(element2radius["C"] + element2radius["H"])
            radii.append(element2radius["C"] + element2radius["H"])
            acceptor_idx.append(acceptor_posi)
            acceptor_idx.append(acceptor_posi)
            hydrogen_idx.append(previous_atom_posi_N)
            hydrogen_idx.append(previous_atom_posi_CB)

        # * The following operations add hydrogens on the C-terminal
        # of dipeptide, proline HAS specific issue here
        if prot.arrays["residuenames"][dipeptide_idx[-1]].strip() == "GLY":
            # * the last residue in the dipeptide is a GLY (glycine),
            # so only 1 hydrogen should be added on it,
            # (CA-C) is replaced by CA-H
            acceptor_posi = residue_match("CA", fidx_match_p_3, -1, 3)
            previous_atom_posi = residue_match("C", fidx_match_p_3, -1, 3)

            # * Set the indices of the additional hydrogen (CA-C)
            radii.append(element2radius["C"] + element2radius["H"])
            acceptor_idx.append(acceptor_posi)
            hydrogen_idx.append(previous_atom_posi)

        elif prot.arrays["residuenames"][dipeptide_idx[-1]].strip() == "PRO":
            # * the last residue in the dipeptide is a PRO (proline),
            # so 3 hydrogens should be added on it,
            # (CA-C) is replaced by CA-H, (CA-CB) is replaced by CA-H,
            # (N-CD) is replaced by N-H
            acceptor_posi = residue_match("CA", fidx_match_p_3, -1, 3)
            previous_atom_posi_C = residue_match("C", fidx_match_p_3, -1, 3)
            previous_atom_posi_CB = residue_match("CB", fidx_match_p_3, -1, 3)

            # * Set the indices of the additional hydrogen (CA-C)
            # * Set the indices of the additional hydrogen (CA-CB)
            radii.append(element2radius["C"] + element2radius["H"])
            radii.append(element2radius["C"] + element2radius["H"])
            acceptor_idx.append(acceptor_posi)
            acceptor_idx.append(acceptor_posi)
            hydrogen_idx.append(previous_atom_posi_C)
            hydrogen_idx.append(previous_atom_posi_CB)

            acceptor_posi_N = residue_match("N", fidx_match_p_3, -1, 3)
            previous_atom_posi_CD = residue_match("CD", fidx_match_p_3, -1, 3)

            # * Set the indices of the additional hydrogen (N-CD)
            radii.append(element2radius["N"] + element2radius["H"])
            acceptor_idx.append(acceptor_posi_N)
            hydrogen_idx.append(previous_atom_posi_CD)

        elif prot.arrays["residuenames"][dipeptide_idx[-1]].strip() != "NME":
            # * the last residue in the dipeptide is NOT a GLY,
            # NOT a PRO (proline) and not an NME
            # (hydrogens should not be added for NME),
            # so 2 hydrogens should be added on it,
            # (CA-C) is replaced by CA-H,
            # (CA-CB) is replaced by CA-H
            acceptor_posi = residue_match("CA", fidx_match_p_3, -1, 3)
            previous_atom_posi_C = residue_match("C", fidx_match_p_3, -1, 3)
            previous_atom_posi_CB = residue_match("CB", fidx_match_p_3, -1, 3)

            # * Set the position of the additional hydrogen (CA-C)
            # * Set the position of the additional hydrogen (CA-CB)
            radii.append(element2radius["C"] + element2radius["H"])
            radii.append(element2radius["C"] + element2radius["H"])
            acceptor_idx.append(acceptor_posi)
            acceptor_idx.append(acceptor_posi)
            hydrogen_idx.append(previous_atom_posi_C)
            hydrogen_idx.append(previous_atom_posi_CB)

        radii = np.array(radii, dtype=np.float32).reshape(-1, 1)
        acceptor_idx = np.array(acceptor_idx)
        hydrogen_idx = np.array(hydrogen_idx)

        return radii, acceptor_idx, hydrogen_idx

    @staticmethod
    def calculate_permutation_indices(
        prot: Atoms, idx: int, dipeptide_length: int
    ) -> tuple[np.ndarray, list[int]]:
        r"""
        Calculate permutation indices to reorder the input atoms. The
        additional hydrogen atoms are assumed to be concatenated to the end of
        the list of the original atoms.

        Parameters:
        -----------
            prot: Atoms
                The protein to be fragmented.
            idx: int
                The index of the dipeptide
            dipeptide_length: int
                The length of the dipeptide with added hydrogen atoms

        Returns:
        --------
            permute_idx: np.ndarray
                The permutation indices for reordering the original and
                additional hydrogen atoms.
            constrain_idx: list[int]
                The indices of the added hydrogen atoms, in the final sequence
                of atoms.

        """
        resi_name, resi_state, next_resi_name, last_resi_name = prot.resi_info[idx]

        atom_index = list(range(dipeptide_length))

        atom_idx = []
        constrain_idx = []  # 注意限制list按照Amber编号，从1开始

        if resi_state == 0:
            # * N-terminus
            # * ----------

            if next_resi_name != "PRO":
                # 按顺序放上即可，新加的氢在最后
                # * ----------------------------
                atom_idx.extend(atom_index[:])

                if next_resi_name != "GLY":
                    # 下一个不是GLY，此时优化最后两个原子
                    constrain_idx.append(dipeptide_length - 2)
                    constrain_idx.append(dipeptide_length - 1)
                else:
                    # 下一个是GLY，此时优化最后一个原子
                    constrain_idx.append(dipeptide_length - 1)

            else:
                # 下一个是PRO，最后一个H应该是加在整个dipeptide的-5号原子的位置
                # * -----------------------------------------------------------
                atom_idx.extend(atom_index[:-5])
                # 将最后一个H放在-5的位置
                atom_idx.append(atom_index[-1])
                constrain_idx.append(dipeptide_length - 5)
                # 写剩下的原子，包括剩下的两个新加的H
                atom_idx.extend(atom_index[-5:-1])
                constrain_idx.append(dipeptide_length - 2)
                constrain_idx.append(dipeptide_length - 1)

        elif resi_state == 1:
            # * C-terminus
            # * ----------

            if last_resi_name != "GLY":
                # 上一个不是GLY的时候，最后两个H来自N端，应该加在C端的2和3两个原子的位置
                # * --------------------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的两个H
                atom_idx.append(atom_index[-2])
                atom_idx.append(atom_index[-1])
                constrain_idx.append(2)
                constrain_idx.append(3)
                # 写剩下的原子
                atom_idx.extend(atom_index[2:-2])

            else:
                # 上一个是GLY的时候，最后一个H来自N端，应该加在C端的2号原子的位置
                # * -------------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的H
                atom_idx.append(atom_index[-1])
                constrain_idx.append(2)
                # 写剩下的原子
                atom_idx.extend(atom_index[2:-1])

        else:
            # * 中间的dipeptide
            # * ---------------

            if (
                last_resi_name != "GLY" and next_resi_name != "GLY" and next_resi_name != "PRO"
            ):
                # 前后没有GLY和PRO，由于整个过程先加N端，再加C端，
                # 此时倒数第三、四个H应该在N端（第三、四个原子），倒数第一、二个原子保持原有位置
                # 注意中间的dipeptide先写C，再写H，和ACE的是反过来的，所以这里是1,0
                # * ----------------------------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的前两个H
                atom_idx.append(atom_index[-3])
                atom_idx.append(atom_index[-4])
                constrain_idx.append(2)
                constrain_idx.append(3)
                # 写中间的原子
                atom_idx.extend(atom_index[2:-4])
                # 优化新加的最后两个H
                atom_idx.extend(atom_index[-2:])
                constrain_idx.append(dipeptide_length - 2)
                constrain_idx.append(dipeptide_length - 1)

            elif (
                last_resi_name != "GLY" and next_resi_name == "GLY"
            ):
                # 下一个是GLY，C端只优化最后的一个原子
                # 此时倒数第二、三个H应该在N端（第三、四个原子），倒数第1个原子保持原有位置
                # * -----------------------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的前两个H
                atom_idx.append(atom_index[-2])
                atom_idx.append(atom_index[-3])
                constrain_idx.append(2)
                constrain_idx.append(3)
                # 写中间的原子
                atom_idx.extend(atom_index[2:-3])
                # 优化新加的最后一个H
                atom_idx.append(atom_index[-1])
                constrain_idx.append(dipeptide_length - 1)

            elif (
                last_resi_name != "GLY" and next_resi_name == "PRO"
            ):
                # 前面不是GLY，后面是PRO，倒数第四、五个H应该在N端（第三、四个原子）
                # 最后一个H加在整个dipeptide的-5号原子的位置
                # * ---------------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的两个H
                atom_idx.append(atom_index[-4])
                atom_idx.append(atom_index[-5])
                constrain_idx.append(2)
                constrain_idx.append(3)
                # 写中间的原子
                atom_idx.extend(atom_index[2:-7])
                # 将最后一个H放在-5的位置
                atom_idx.append(atom_index[-1])
                constrain_idx.append(dipeptide_length - 5)
                # 写剩下的原子
                atom_idx.extend(atom_index[-7:-5])
                # 优化剩下的新加的两个H
                atom_idx.extend(atom_index[-3:-1])
                constrain_idx.append(dipeptide_length - 2)
                constrain_idx.append(dipeptide_length - 1)

            elif (
                last_resi_name == "GLY" and next_resi_name != "GLY" and next_resi_name != "PRO"
            ):
                # 前面是GLY，后面不是GLY也不是PRO，倒数第三个H应该在N端（第三个原子）
                # 倒数第一、二个原子保持原有位置
                # * ----------------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的第一个H
                atom_idx.append(atom_index[-3])
                constrain_idx.append(2)
                # 写中间的原子
                atom_idx.extend(atom_index[2:-3])
                # 优化新加的最后两个H
                atom_idx.extend(atom_index[-2:])
                constrain_idx.append(dipeptide_length - 2)
                constrain_idx.append(dipeptide_length - 1)

            elif (
                last_resi_name == "GLY" and next_resi_name == "GLY"
            ):
                # 前面是GLY，后面也是GLY，倒数第二个H应该在N端（第三个原子）
                # 倒数第一个原子保持原有位置
                # * -------------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的第一个H
                atom_idx.append(atom_index[-2])
                constrain_idx.append(2)
                # 写中间的原子
                atom_idx.extend(atom_index[2:-2])
                # 优化新加的最后一个H
                atom_idx.append(atom_index[-1])
                constrain_idx.append(dipeptide_length - 1)

            elif (
                last_resi_name == "GLY" and next_resi_name == "PRO"
            ):
                # 前面是GLY，后面是PRO，倒数第四个H应该在N端（第三个原子）
                # 最后一个H加在整个dipeptide的-5号原子的位置
                # * -----------------------------------------------------
                atom_idx.append(atom_index[1])
                atom_idx.append(atom_index[0])
                # 优化新加的第一个H(倒数第四个H)
                atom_idx.append(atom_index[-4])
                constrain_idx.append(2)
                # 写中间的原子
                atom_idx.extend(atom_index[2:-6])
                # 将最后一个H放在-5的位置
                atom_idx.append(atom_index[-1])
                constrain_idx.append(dipeptide_length - 5)
                # 写剩下的原子
                atom_idx.extend(atom_index[-6:-4])
                # 优化剩下的新加的两个H
                atom_idx.extend(atom_index[-3:-1])
                constrain_idx.append(dipeptide_length - 2)
                constrain_idx.append(dipeptide_length - 1)

            else:
                raise ValueError(
                    f"Unhandled combination of last residue: {last_resi_name}"
                    f" and next residue: {next_resi_name}"
                )

        with gzip.open(Preprocess.get_seq_dict_path()) as f:
            sequence_dict = pickle.load(f)

        combination = sequence_dict[f"{last_resi_name}_{resi_name}_{next_resi_name}"]
        permute_idx = np.array(atom_idx, dtype=int)[list(combination.keys())]

        return permute_idx, constrain_idx

    @staticmethod
    def calculate_select_indices(
        permute_index: list[np.ndarray], constrain_index: list[list[int]]
    ) -> list[np.ndarray]:
        r"""
        Calculate the indices in the fragments that correspond to the original
        atoms in the protein. This is required to extract the forces that
        correspond to the original atoms, i.e. exclude the forces on the added
        hydrogens. This is a concatenated list of both dipeptides and ACE-NMEs.

        Parameters:
        -----------
            permute_index: list[np.ndarray]
                The list of indices that represent the permutation from a
                simple concatenation of original atoms and added hydrogens to a
                proper ordering.
            constrain_index: list[list[int]]
                The list of indices of atoms in the dipeptides (after
                permutation) that are the added hydrogens

        Returns:
        --------
            select_index: list[list[int]]
                The indices in the fragments that correspond to the original
                atoms (excluding the added Hs). range: [0, length)

        """

        select_index = []

        # calculate indices for dipeptides: invert permutation and exclude
        # hydrogens (added at the end)
        for permute_idx, constrain_idx in zip(permute_index, constrain_index):
            length = len(permute_idx)

            select_idx = np.empty((length), dtype=int)
            select_idx[permute_idx] = np.arange(length)
            select_idx = select_idx[:-len(constrain_idx)]

            select_index.append(select_idx)

        # itertools.pairwise (available in 3.10)
        def pairwise(iterable):
            iterator = iter(iterable)
            one = next(iterator, None)
            for two in iterator:
                yield one, two
                one = two

        # mask indices of added hydrogens
        masked_index = [np.copy(v) for v in permute_index]
        for masked_idx, constrain_idx in zip(masked_index, constrain_index):
            masked_idx[constrain_idx] = np.iinfo(int).max

        # calculate indices for acenmes: extract unmasked indices and invert
        # permutation
        for nme, ace in pairwise(masked_index):
            acenme_idx = np.concatenate([ace[:6], nme[-6:]], axis=0)
            count = np.count_nonzero(acenme_idx != np.iinfo(int).max)
            select_idx = np.argsort(acenme_idx, axis=0)[:count]

            select_index.append(select_idx)

        return select_index

    @staticmethod
    def get_cystine_bonds(prot: Protein, dipeptides_index: list[list[int]]):
        atom_types = prot.arrays["atomtypes"]
        residuenames = [name.strip() for name in prot.arrays["residuenames"]]

        dipeptide_CYX_idx = []
        atom_S_idx = []

        for idx, dipeptide_idx in enumerate(dipeptides_index):
            if residuenames[dipeptide_idx[6]] != "CYX":
                continue

            dipeptide_CYX_idx.append(idx)

            for atom_idx in dipeptide_idx:
                if atom_types[atom_idx] != "SG":
                    continue

                atom_S_idx.append(atom_idx)

        assert len(dipeptide_CYX_idx) == len(atom_S_idx), "inconsistent CYX, SG counts!"
        assert len(dipeptide_CYX_idx) % 2 == 0, "odd number of CYX residues!"

        if not len(dipeptide_CYX_idx):
            return {}

        if arguments.get().verbose >= 1:
            print(f" [i] {len(dipeptide_CYX_idx)} CYX residues found")

        S_positions = prot.get_positions()[atom_S_idx]
        S_distances = np.linalg.norm(S_positions[np.newaxis, :] - S_positions[:, np.newaxis], axis=-1)
        np.fill_diagonal(S_distances, np.inf)

        S_pairs = {}
        for i, j in enumerate(np.argmin(S_distances, axis=-1)):
            if i in S_pairs or j in S_pairs:
                continue

            S_pairs[i] = j

        return {dipeptide_CYX_idx[i]: dipeptide_CYX_idx[j] for i, j in S_pairs.items()}

    @staticmethod
    def create_protein_graph(
        resi_info: list[Any], dipeptides_length: list[int], constrain_index: list[list[int]]
    ) -> list[ProteinData]:
        r"""
        Create the protein data structure (torch_geometric.data.Data) used for
        the optimisation of the positions of the added hydrogens in the
        dipeptides.

        Parameters:
        -----------
            resi_info: list[Any]
                The list containing residue info
            dipeptides_length: list[int]
                The lengths of the dipeptides with added hydrogen atoms
            constrain_index: list[list[int]]
                The list of indices of atoms in the dipeptides (after
                permutation) that are the added hydrogens

        Returns:
        --------
            protein_data: list[ProteinData]
                The list of Data objects that correspond to the dipeptides.

        """
        fpath = os.path.dirname(__file__)
        ctables = {
            res_name: CTable.from_prmtop(f"{fpath}/prmtop/{info[0]}.prmtop")
            for res_name, info in fragment_info.items()
        }

        protein_data = []

        # atom_idx: indices of atoms to optimise (added hydrogens)
        # other_idx: indices of atoms with fixed positions (original atoms)
        for info, length, constrain_idx in zip(resi_info, dipeptides_length, constrain_index):
            if not length:
                continue

            atom_idx = torch.tensor(constrain_idx, dtype=torch.long)
            size_idx = torch.arange(length, dtype=torch.long)
            mask_idx = torch.zeros((length,), dtype=torch.long)
            mask_idx = mask_idx.scatter(dim=0, index=atom_idx, value=1)
            other_idx = size_idx[~mask_idx.to(dtype=torch.bool)]
            positions = torch.zeros([length, 3])
            ctable = ctables[info[0]]

            protein_data.append(ProteinData(atom_idx, other_idx, positions, ctable))

        return protein_data
