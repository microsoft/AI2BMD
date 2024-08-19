from functools import lru_cache

import numpy as np
from ase import Atoms


class FragmentData:
    def __init__(self, z: np.ndarray, pos: np.ndarray, sym: str, start: np.ndarray, end: np.ndarray, batch: np.ndarray):
        self.z = z
        self.pos = pos
        self.sym = sym
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
            self.sym[f_idx],
            self.start[f_idx] - self.start[f_idx.start],
            self.end[f_idx] - self.start[f_idx.start],
            self.batch[a_idx] - self.batch[a_idx.start],
        )

    @lru_cache
    def scalar_split(self):
        return np.arange(0, len(self), 2, dtype=int), np.arange(1, len(self), 2, dtype=int)

    @lru_cache
    def vector_split(self):
        return (self.batch % 2 == 0), (self.batch % 2 == 1)

    def __len__(self):
        return len(self.start)

    def get_atoms(self, idx: int) -> Atoms:
        start, end = self.start[idx], self.end[idx]

        return Atoms(numbers=self.z[start:end], positions=self.pos[start:end])


class FragmentInfo:
    @classmethod
    def split(cls, total):
        return (total + 1) // 2, total // 2
