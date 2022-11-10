from multiprocessing import Pool
from rdkit.Chem import AllChem
from rdkit import Chem 
from ogb.lsc import PCQM4Mv2Dataset
from tqdm import tqdm
import pickle
import sys
import os
import numpy as np
from compound_tools import mol_to_trans_data_w_rdkit3d
from multiprocessing import Pool

DATA_ROOT, SAVE_ROOT, TC, machine_idx, sub_length = sys.argv[1:]
machine_idx = int(machine_idx)
sub_length = int(sub_length)

def ReorderAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum

def get_atom_poses(mol):
    atom_poses = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    return atom_poses

def worker(data):
    
    smiles, label, sdf_mol = data
    return_data = {}
    
    mol = AllChem.MolFromSmiles(smiles)
    mol = ReorderAtoms(mol)

    return_data.update(mol_to_trans_data_w_rdkit3d(mol))
    return_data['rdkit_atom_pos'] = return_data.pop('atom_pos')
    return_data['label'] = label
    
    if sdf_mol is not None:
        sdf_mol = ReorderAtoms(sdf_mol)
        sdf_atom_list = np.array([atom.GetAtomicNum() for atom in sdf_mol.GetAtoms()]) 
        smiles_atom_list = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        assert np.all(sdf_atom_list == smiles_atom_list)
        return_data['eq_atom_pos'] = np.array(get_atom_poses(sdf_mol))
    else:
        return_data['eq_atom_pos'] = None

    return return_data

def smiles2graph(dataset, machine_idx):
    
    data_list = []
    with Pool(120) as pool:
        iter = pool.imap(worker, dataset)
        
        for i, data in tqdm(enumerate(iter), total=len(dataset), desc='Converting'):
            data_list.append(data)
            
    with open(os.path.join(SAVE_ROOT, f"{machine_idx}_data_list.pickle"), "wb") as f:
        pickle.dump(data_list, f)
        
        
def load_data(root, test_challenge="False"):

    suppl = Chem.SDMolSupplier(os.path.join(DATA_ROOT, 'pcqm4m-v2-train.sdf'), removeHs=True)
    dataset = PCQM4Mv2Dataset(root=root, only_smiles=True)

    dataset_list = []
    if test_challenge == "False":
        # process all dataset
        print("Process all dataset")
        dataset_idx = range(len(dataset))
    else:
        # only process the test challenge split
        print("Process test challenge split")
        dataset_idx = dataset.get_idx_split()["test-challenge"]
        
    for idx in tqdm(dataset_idx, desc="Loading data"):
        smiles, label = dataset[idx]
        if idx >= len(suppl):  
            dataset_list.append([smiles, label, None])
        else:
            dataset_list.append([smiles, label, suppl[idx]])
        
    return dataset_list

def slice_dataset(dataset, ids, ide):
    
    slices = [i for i in range(ids, ide)]
    data_list = []
    for idx in slices:
        data_list.append(dataset[idx])
        
    return data_list

if __name__ == "__main__":
    
    dataset = load_data(root=DATA_ROOT, test_challenge=TC)
    slice_data = slice_dataset(dataset, machine_idx * sub_length, (machine_idx + 1) * sub_length if (machine_idx + 1) * sub_length < len(dataset) else len(dataset))
    os.makedirs(SAVE_ROOT, exist_ok=True)
    smiles2graph(slice_data, machine_idx)
        
