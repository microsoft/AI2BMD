import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from os import path as osp
from typing import Any, Callable, List

import numpy as np
import torch
from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.md.md import MolecularDynamics

from AIMD import arguments
from utils.system import get_physical_core_count


def read_protein(fpath: str) -> Atoms:
    r"""
    Convert .pdb file to ase Atoms object.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    atoms = read(fpath)
    args = arguments.get()
    if len(atoms) == 0:
        raise ValueError("Error: The PDB file is empty!")
    
    return atoms


def fill_atom_symbol(atom_symbol: str) -> str:
    if len(atom_symbol) > 4:
        raise ValueError("atom symbol is too long")
    elif len(atom_symbol) == 4:
        return atom_symbol
    elif 0 < len(atom_symbol) < 4:
        return " " + atom_symbol + " " * (3 - len(atom_symbol))
    elif len(atom_symbol) <= 0:
        raise ValueError("atom symbol is too short")
    return ""


def reorder_atoms(fpath: str):
    r"""
    Reorder atoms in .pdb output from tinker.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    with open(fpath, 'r') as f:
        lines = f.readlines()

    output = []
    sidechain = []

    res_id = None
    res_count = 0
    h_found = False

    for l in lines:
        cols = l.split()

        if len(cols) < 8 or cols[0] != 'ATOM':
            output.append(l)
            continue

        if cols[4] != res_id:
            res_count = 0

        # check if sidechain atoms should be written
        if cols[2] == 'H' or cols[2] == 'HA':
            res_count = 0
            h_found = True
        elif h_found is True:
            # write sidechain atoms right after H/HA
            output.extend(sidechain)
            sidechain = []
            h_found = False

        # enumerate N/CA/C/O atoms
        if res_count == 0 and cols[2] == 'N':
            res_count += 1
        elif res_count == 1 and cols[2] == 'CA':
            res_count += 1
        elif res_count == 2 and cols[2] == 'C':
            res_count += 1
        elif res_count == 3 and cols[2] == 'O':
            res_count += 1
        # save rows between N/CA/C/O and H/HA
        elif res_count == 4:
            sidechain.append(l)
            res_id = cols[4]
            continue

        # save line to output
        output.append(l)

        # update current residue id
        res_id = cols[4]

    with open(fpath, 'w') as f:
        for l in output:
            f.write(l)


def get_residue_name(prot: Atoms) -> List[str]:
    atom_resname: List[str] = prot.arrays["residuenames"].tolist()
    atom_resid: List[np.int64] = prot.arrays["residuenumbers"].tolist()
    mol_resid = list(set(atom_resid))
    mol_resid.sort(key=atom_resid.index)
    mol_index = [atom_resid.index(id) for id in mol_resid]
    mol_resname = [
        atom_resname[index].strip()
        for index in mol_index
        if atom_resname[index].strip() != "WAT"
    ]
    return mol_resname


def record_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} takes {end - start} seconds")
        return result

    return wrapper


class RNGPool:
    def __init__(self, seed, shape, count):
        self.rng = np.random.default_rng(seed)

        self.pool = deque()
        self.shape = shape
        self.count = count

        self.fill()

    def fill(self):
        while len(self.pool) < self.count:
            self.pool.append(self.rng.standard_normal(self.shape))

    def drain(self):
        return self.pool.popleft()

    def standard_normal(self, size):
        if size == self.shape and len(self.pool):
            return self.drain()
        else:
            return self.rng.standard_normal(size)


class SkipCheckState:
    """Temporarily disables atoms.check_state so that it does not compare"""
    def __init__(self, atoms):
        self.skip_check_state = getattr(atoms, 'skip_check_state', False)
        self.atoms = atoms
        atoms.skip_check_state = True

    def __enter__(self, *_):
        pass

    def __exit__(self, *_):
        self.atoms.skip_check_state = self.skip_check_state

_fragment_step_time = None

_workqueue_instance = None

class WorkQueue():
    def __init__(self):
        global _workqueue_instance
        self.work: deque[Callable] = deque()
        if _workqueue_instance is not None:
            raise RuntimeError("There should be only one WorkQueue instance.")
        _workqueue_instance = self

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.work)

    def submit(self, action):
        return self.work.append(action)

    def drain(self):
        while len(self.work):
            self.work.popleft()()

    @classmethod
    def finalise(cls):
        if _workqueue_instance:
            _workqueue_instance.drain()


def delay_work(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if _workqueue_instance:
            _workqueue_instance.submit(partial(f, *args, **kwargs))
        else:
            # root calc isn't workqueue-aware
            # do not delay the work because nobody will drain the works
            f(*args, **kwargs)

    return wrapper


class TemperatureRunawayError(RuntimeError):
    def __init__(self, temp_k: int, *args: object):
        self.temp = temp_k
        super().__init__(args)


class MDObserver:
    """An observer class, offering functions that can be attached to ASE MolecularDynamics object, notified at every step."""

    def __init__(self, a: Atoms, md: MolecularDynamics, traj: TrajectoryWriter, rng: RNGPool, step_offset: int, temp_k: int):
        self.a = a
        self.md = md
        self.traj = traj
        self.rng = rng
        self.step_offset = step_offset
        self.prev_step_time = None
        self.copy = None
        self.temp_k = temp_k

    def get_md_step(self):
        return self.step_offset + self.md.nsteps

    def save_traj_copy(self):
        self.copy = self.a.copy()

    @delay_work
    def write_traj(self):
        with SkipCheckState(self.copy):
            self.traj.write(self.copy)

    def printenergy(self):
        """
        Function to print the potential, kinetic and total energy
        """
        # per atom need / len(a)

        with SkipCheckState(self.a):
            cur_time = time.perf_counter()
            epot = self.a.get_potential_energy().item()
            ekin = self.a.get_kinetic_energy().item()
            temperature = self.a.get_temperature()
            if temperature > 1.5*self.temp_k:
                raise TemperatureRunawayError(temperature, "temperature runaway")
            if self.prev_step_time is None:
                steptime = ""
            else:
                steptime = f"  time = {(cur_time - self.prev_step_time) * 1000:.1f}ms"
            if _fragment_step_time is None:
                frag_steptime = ""
            else:
                frag_steptime = f"  fragtime = {_fragment_step_time * 1000:.1f}ms"
            self.prev_step_time = cur_time
            print(f"Step {self.get_md_step():d}: "
                  f"Epot = {epot:.3f}eV  "
                  f"Ekin = {ekin:.3f}eV "
                  f"(T = {temperature:3.0f}K) "
                  f"Etot = {epot+ekin:.3f}eV"
                  f"{steptime}"
                  f"{frag_steptime}")

    @delay_work
    def fill_rng_pool(self):
        """
        Function to fill the RNG pool
        """
        self.rng.fill()


class PDBAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.covalent_radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "P": 1.07,
            "S": 1.05,
        }
        self.atoms = self.parse_pdb()

    def parse_pdb(self):
        """Parse a PDB file and return a list of atoms and their coordinates."""
        atoms = []
        with open(self.filename, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append((atom_name, np.array([x, y, z])))
        return atoms

    def compute_distance(self, atom1, atom2):
        """Compute the Euclidean distance between two atoms."""
        _, (_, pos1) = atom1
        _, (_, pos2) = atom2
        return np.linalg.norm(pos1 - pos2)

    def find_bonded_atoms(self, target_atom_name):
        """Find atoms that are bonded to a target atom type based on distance."""
        covalent_radius = self.covalent_radii.get(target_atom_name, 0)
        indexed_atoms = [
            (i, atom)
            for i, atom in enumerate(self.atoms)
            if atom[0].startswith(target_atom_name)
        ]
        bonded_atoms = []
        for i1, atom1 in indexed_atoms:
            for i2, atom2 in enumerate(self.atoms):
                if i1 != i2:
                    distance = self.compute_distance((i1, atom1), (i2, atom2))
                    atom2_radius = self.covalent_radii.get(atom2[0][0], 0)
                    idea_length = covalent_radius + atom2_radius
                    if distance <= idea_length + 0.2:
                        bonded_atoms.append((i1, i2, idea_length + 0.2, 15))
        assert len(indexed_atoms) == len(
            bonded_atoms
        ), "Hydrogen constraint: hydrogen covalent bonds != hydrogen num"
        return bonded_atoms


kcalmol2ev = 1 * (units.kcal / units.mol) / units.eV


def src_dir():
    return osp.abspath(osp.join(osp.dirname(__file__), ".."))


# helpers for common numpy -> torch operations
def numpy_to_torch(x: torch.Tensor, device: str):
    return torch.from_numpy(x).to(device)

def numpy_list_to_torch(x: list[torch.Tensor], device: str):
    return torch.from_numpy(np.concatenate(x)).to(device)


# wrapper for serial/parallel execution
def execution_wrapper(f_args: list[Any], concurrent: bool):
    if concurrent is True:
        with ThreadPoolExecutor(len(f_args)) as executor:
            futures = [executor.submit(f, *args) for f, *args in f_args]

        return [f.result() for f in futures]
    else:
        return [f(*args) for f, *args in f_args]

