import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.units import Hartree
import subprocess
from pyscf import dft, gto
from pyscf.geomopt.geometric_solver import optimize
from pyscf import lib
from typing import List
from ogb.lsc import PCQM4Mv2Evaluator


"""All the smiles whose number of atom is smaller than 4 will be post-processed since ViSNet models
dihedral information, while these smiles don't provide dihedral aninformation.

If the smiles appear in train set or valid set only once or multiple times with the std of these gaps is 0, 
then these gaps will be read from `intersect_results`.
"""


def postprocess(csv_file: str='data.csv.gz', split_file: str='split_dict.pt'):
    """The post process script.
    """
    calc_smiles = filter_ogb(csv_file, split_file)
    intersect_results = intersect_smiles(csv_file, split_file)

    gaps = {}
    for smiles in tqdm(calc_smiles, desc='Postprocess'):
        if intersect_results.get(smiles) is not None:
            if len(intersect_results.get(smiles)) == 1 or (len(intersect_results.get(smiles)) > 1 and intersect_results.get(smiles).std() == 0):
                gaps[smiles] = intersect_results.get(smiles).mean()
                continue
            
        try:
            atom, symbols = generate_3d(smiles)
            gap = pyscf_calc(atom, symbols, optimize_=True if len(symbols) > 1 else False)
            gaps[smiles] = gap
        except:
            print(f'The 3D conformation of {smiles} failed to generate! Skip this molecule!')
    return gaps

def rewrite(pred, replaced_gaps, csv_file: str='data.csv.gz', split_file: str='split_dict.pt'):
    """Rewrite the submission output with the post-processed results.
    """
    df = pd.read_csv(csv_file)
    smiles_list = df['smiles']

    split = torch.load(split_file)
    test_smiles = smiles_list.values[split['test-challenge']]
    for i in replaced_gaps.keys():
        pred[np.where(test_smiles == i)] = [replaced_gaps[i]] * len(np.where(test_smiles == i)[0])

    return pred


def pyscf_calc(atom: str, symbols: List[str], optimize_: bool=False):
    """Optimization or scf calculation with `pyscf`. If the number of atom is 1, perform scf calculation directly.
    """
    with lib.with_omp_threads(48):
        mol = gto.M(atom=atom, basis='6-31g*')
    
        rks = dft.RKS(mol)
    
        rks.xc = 'b3lyp'
    
        if optimize_:
            mol_eq = optimize(rks, maxsteps=100)
    
            atom = '; '.join([symbols[i] + '  ' + ' '.join(map(str, mol_eq.atom_coords()[i].tolist())) for i in range(len(symbols))])
            mol_eq = gto.M(atom=atom, basis='6-31g*')
            mol_eq.unit = 'B'  # The unit of optimized molecule is Bohr
            mol_eq.build()
    
            rks_eq = dft.RKS(mol_eq)
            rks_eq.xc = 'b3lyp'
    
            rks_eq.kernel()
            eigens = rks_eq.mo_energy
    
            occs = rks_eq.get_occ()
        else:
            rks.kernel()
            eigens = rks.mo_energy
            occs = rks.get_occ()
    
        occ = np.where(occs != 0)[0].max()
    
        gap = (eigens[occ+1] - eigens[occ]) * Hartree
        return gap


def generate_3d(smiles: str):
    """Generate 3D structures with `RDKit`.
    """
    mol_ = Chem.MolFromSmiles(smiles)
    mol_ = Chem.AddHs(mol_)
    ret = AllChem.EmbedMolecule(mol_, randomSeed=0xf00d)

    if ret == -1:
        mol_ = Chem.RemoveHs(mol_)
        smiles = Chem.MolToSmiles(mol_)
        mol_ = Chem.MolFromSmiles(smiles)
        mol_ = Chem.AddHs(mol_)

        ret = AllChem.EmbedMolecule(mol_, randomSeed=0xf00d)

    ret = AllChem.UFFOptimizeMolecule(mol_)
    assert ret >= 0

    infos = Chem.MolToMolBlock(mol_).split('\n')
    num_atoms = mol_.GetNumAtoms()

    context = ''
    symbols = []

    for line in infos[4:4+num_atoms]:
        tmp = line.split()
        context += tmp[3] + '  ' + ' '.join(tmp[:3]) + ';'
        symbols.append(tmp[3])

    return context[:-1], symbols


def filter_ogb(csv_file: str='data.csv.gz', split_file: str='split_dict.pt'):
    """Find all the smiles in the test challenge dataset, whose number of atom is smaller than 4.
    Besides, find all the smiles that exists elements don't appear in train set and valid set. 
    """
    df = pd.read_csv(csv_file)
    smiles_list = df['smiles']

    split = torch.load(split_file)
    test_smiles = smiles_list.values[split['test-challenge']]

    calc_smiles = []
    elements_test = set()
    for smiles in tqdm(test_smiles, desc='Molecules with 1-4 atoms in test set'):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() < 4:
            calc_smiles.append(smiles)

        for i in mol.GetAtoms():
            elements_test.add(i.GetSymbol())

    # elements don't appear in train set and valid set
    elements_train_val = set()
    for smiles in tqdm(smiles_list.values[np.array(split['train'].tolist() + split['valid'].tolist())], desc='Elements absent in train and val set'):
        mol_ = Chem.MolFromSmiles(smiles)
        mol_ = Chem.AddHs(mol_)
        for i in mol_.GetAtoms():
            elements_train_val.add(i.GetSymbol())

    elements_missed = elements_test.difference(elements_train_val)
    for smiles in tqdm(test_smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        for i in mol.GetAtoms():
            if i.GetSymbol() in elements_missed:
                calc_smiles.append(smiles)
                break

    return set(calc_smiles)


def intersect_smiles(csv_file: str='data.csv.gz', split_file: str='split_dict.pt'):
    """Return a dict whose keys are smiles that appear both in trainset (or valid set) and test challenge set, 
    and values are the gaps of these smiles in train set and valid set.
    """
    df = pd.read_csv(csv_file)
    smiles_list = df['smiles']
    gaps = df['homolumogap'].values

    split = torch.load(split_file)
    test_smiles = smiles_list.values[split['test-challenge']]

    trn_val_idx = np.array(split['train'].tolist() + split['valid'].tolist())
    train_val_smiles = df['smiles'].values[trn_val_idx]

    intersect_smiles = set(train_val_smiles.tolist()).intersection(set(test_smiles.tolist()))

    intersect_results = {}
    for i in tqdm(intersect_smiles, desc='Record repeat smiles'):
        intersect_results[i] = gaps[trn_val_idx[np.where(train_val_smiles == i)[0]]]

    return intersect_results


def read_xyz(xyz_file: str):
    """Read xyz file and get the input for pyscf
    """
    with open(xyz_file, 'r') as f:
       num_atoms = int(f.readline())
       f.readline()

       symbols, pos = [], []
       for _ in range(num_atoms):
           info = f.readline()
           tmp = info.split()
           symbols.append(tmp[0])
           pos.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])

       atom = '; '.join([symbols[i] + '  ' + ' '.join(map(str, pos[i])) for i in range(len(symbols))])

    return atom, symbols