from functools import lru_cache

import numpy as np
from ase import Atoms


class FragmentData:
    def __init__(self, z: np.ndarray, pos: np.ndarray, start: np.ndarray, end: np.ndarray, batch: np.ndarray):
        self.z = z
        self.pos = pos
        self.start = start
        self.end = end
        self.batch = batch

    def __getitem__(self, f_idx):
        # f_idx: index of fragment [0, __len__)
        if isinstance(f_idx, int):
            f_idx = slice(f_idx, f_idx + 1)

        # a_idx: index of atoms [0, end[-1])
        a_idx = slice(self.start[f_idx][0], self.end[f_idx][-1])

        return FragmentData(
            self.z[a_idx],
            self.pos[a_idx],
            self.start[f_idx] - self.start[f_idx.start],
            self.end[f_idx] - self.start[f_idx.start],
            self.batch[a_idx] - self.batch[a_idx.start],
        )

    @lru_cache
    def scalar_split(self):
        valid = np.flatnonzero(self.end - self.start)
        split = np.zeros(len(self), dtype=int)
        split[0::2] = 1
        split = split[valid]

        return split == 1, split == 0

    @lru_cache
    def vector_split(self):
        split = np.zeros(self.end[-1], dtype=int)
        np.add.at(split, self.start[0::2], 1)
        np.add.at(split, self.start[1::2], -1)
        split = np.cumsum(split, axis=0)

        return split == 1, split == 0

    def __len__(self):
        return len(self.start)

    def get_atoms(self, idx: int) -> Atoms:
        start, end = self.start[idx], self.end[idx]

        return Atoms(numbers=self.z[start:end], positions=self.pos[start:end])


class FragmentInfo:
    @classmethod
    def split(cls, total):
        return (total + 1) // 2, total // 2
