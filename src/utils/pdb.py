import math
import re

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read


def read_protein(fpath: str) -> Atoms:
    r"""
    Convert .pdb file to ase Atoms object.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    atoms = read(fpath)
    if len(atoms) == 0:
        raise ValueError("Error: The PDB file is empty!")

    return atoms


def fix_atomic_numbers(fpath: str, atoms: Atoms):
    r"""
    Fix atomic numbers. ase.io.read interprets atom names incorrectly when the
    atoms are denoted as part of a protein residue.
    """
    symbols = []
    with open(fpath, 'r') as f:
        for l in f.readlines():
            if l.startswith('ATOM') or l.startswith('HETATOM'):
                symbol = l[12:14].strip()
                if symbol.startswith('H'):
                    symbol = 'H'
                symbols.append(symbol)

    _numbers = atoms.get_atomic_numbers()
    _numbers[:len(symbols)] = [chemical_symbols.index(sym) for sym in symbols]

    atoms.set_atomic_numbers(_numbers)


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


def standardise_pdb(fpath: str):
    r"""
    Check and rewrite residue numbers in .pdb output from tinker. Wraps residue
    numbers > 9999 to 0 so that ase can process the .pdb file correctly.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    with open(fpath, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue

            try:
                # ase.io.proteindatabank: extract the residue number
                res_idx = int(line[22:26].split()[0])
            except IndexError:
                break
            else:
                return

    output = []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # wrap the residue number on 10000
                res_idx = int(line[6:].split()[3])
                end = re.search(r'\b({})\b'.format(res_idx), line[22:]).end() + 22
                output.append(line[:22] + f"{res_idx % 10000: >4}" + line[end:])
            else:
                output.append(line)

    with open(fpath, 'w') as f:
        for l in output:
            f.write(l)


# Move pdb coordinates according to the center of mass
def translate_coord_pdb(inpfile: str, outfile: str):
    atomic_masses = {
        'H': 1.008,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'F': 18.998,
        'P': 30.974,
        'S': 32.06,
        'NA': 22.990,
        'CL': 35.453,
    }
    atoms = []
    masses = []
    with open(inpfile, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_type = line[76:78].strip()
                mass = atomic_masses.get(atom_type, 0)
                atoms.append([x, y, z])
                masses.append(mass)
    atoms = np.array(atoms)
    masses = np.array(masses)

    # get mass center
    total_mass = np.sum(masses)
    mass_center = np.sum(atoms * masses[:, np.newaxis], axis=0) / total_mass

    # translate atoms
    atoms -= mass_center

    # write pdb
    with open(inpfile, 'r') as file:
        original_lines = file.readlines()

    # get pdc
    pbc_x = math.ceil(np.max(np.abs(atoms[:, 0])) * 2 + 20)
    pbc_y = math.ceil(np.max(np.abs(atoms[:, 1])) * 2 + 20)
    pbc_z = math.ceil(np.max(np.abs(atoms[:, 2])) * 2 + 20)

    with open(outfile, 'w') as file:
        atom_index = 0
        # file.write(f"HEADER    {pbc_x:.3f} {pbc_y:.3f} {pbc_z:.3f}\n")
        for line in original_lines[1:]:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x, y, z = atoms[atom_index]
                file.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                atom_index += 1
            else:
                file.write(line)

    return pbc_x, pbc_y, pbc_z


def reorder_coord_amber2tinker(fpath: str):
    r"""
    Reorder atoms in .pdb output from tinker.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"

    reorder_dict = {
        'ACE': [1, 4, 5, 0, 2, 3],
        'ALA': [0, 2, 8, 9, 1, 3, 4, 5, 6, 7],
        'ARG': [0, 2, 22, 23, 1, 3, 4, 7, 10, 13, 15, 16, 19, 5, 6, 8, 9, 11, 12, 14, 17, 18, 20, 21],
        'ASN': [0, 2, 12, 13, 1, 3, 4, 7, 8, 9, 5, 6, 10, 11],
        'ASP': [0, 2, 10, 11, 1, 3, 4, 7, 8, 9, 5, 6],
        'CYS': [0, 2, 9, 10, 1, 3, 4, 7, 5, 6, 8],
        'CYX': [0, 2, 8, 9, 1, 3, 4, 7, 5, 6],
        'GLN': [0, 2, 15, 16, 1, 3, 4, 7, 10, 11, 12, 5, 6, 8, 9, 13, 14],
        'GLU': [0, 2, 13, 14, 1, 3, 4, 7, 10, 11, 12, 5, 6, 8, 9],
        'GLY': [0, 2, 5, 6, 1, 3, 4],
        'HIE': [0, 2, 15, 16, 1, 3, 4, 7, 8, 13, 9, 11, 5, 6, 14, 10, 12],
        'ILE': [0, 2, 17, 18, 1, 3, 4, 10, 6, 13, 5, 11, 12, 7, 8, 9, 14, 15, 16],
        'LEU': [0, 2, 17, 18, 1, 3, 4, 7, 9, 13, 5, 6, 8, 10, 11, 12, 14, 15, 16],
        'LYS': [0, 2, 20, 21, 1, 3, 4, 7, 10, 13, 16, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 19],
        'MET': [0, 2, 15, 16, 1, 3, 4, 7, 10, 11, 5, 6, 8, 9, 12, 13, 14],
        'PHE': [0, 2, 18, 19, 1, 3, 4, 7, 8, 16, 10, 14, 12, 5, 6, 9, 17, 11, 15, 13],
        'PRO': [0, 10, 12, 13, 11, 7, 4, 1, 8, 9, 5, 6, 2, 3],
        'SER': [0, 2, 9, 10, 1, 3, 4, 7, 5, 6, 8],
        'THR': [0, 2, 12, 13, 1, 3, 4, 10, 6, 5, 11, 7, 8, 9],
        'TRP': [0, 2, 22, 23, 1, 3, 4, 7, 8, 21, 10, 12, 19, 13, 17, 15, 5, 6, 9, 11, 20, 14, 18, 16],
        'TYR': [0, 2, 19, 20, 1, 3, 4, 7, 8, 17, 10, 15, 12, 13, 5, 6, 9, 18, 11, 16, 14],
        'VAL': [0, 2, 14, 15, 1, 3, 4, 6, 10, 5, 7, 8, 9, 11, 12, 13],
        'NME': [0, 2, 1, 3, 4, 5],
    }

    output = []

    amino_acids = []
    residue_names = ['']
    atom_start = False

    with open(fpath, 'r') as f:
        lines = f.readlines()

        res_idx = None
        atoms = []

        for l in lines:
            cols = l.split()

            if not atom_start and not l.startswith('ATOM') and not l.startswith('HETATM'):
                output.append(l)
                continue

            atom_start = True

            if atom_start and not l.startswith('ATOM') and not l.startswith('HETATM'):
                continue

            if cols[4] != res_idx:
                res_idx = cols[4]

                residue_names.append(cols[3])
                amino_acids.append(atoms)
                atoms = []

            atoms.append(l)

        amino_acids.append(atoms)

    for residue, atoms in zip(residue_names[1:], amino_acids[1:]):
        if residue not in reorder_dict:
            output.extend(atoms)
        else:
            reordered_atoms = [atoms[i] for i in reorder_dict[residue]]
            output.extend(reordered_atoms)

    with open(fpath, 'w') as f:
        for l in output:
            f.write(l)
