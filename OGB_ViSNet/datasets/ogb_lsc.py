import os
import os.path as osp
import shutil
import tarfile
from multiprocessing import Pool
from typing import (Callable, List, Optional, Union, Tuple)
import pickle

from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from OGB_ViSNet.datasets.utils import data2graph, preprocess_item
import pandas as pd
from rdkit import Chem

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from functools import lru_cache

class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=data2graph, AddHs=False, tc=False, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        self.AddHs = AddHs
        self.tc = tc


        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.pos_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        if self.tc:
            return 'geometric_data_H_processed_tc.pt' if self.AddHs else 'geometric_data_processed_tc.pt'
        return 'geometric_data_H_processed.pt' if self.AddHs else 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, 'r:gz')
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print('Stop download')
            exit(-1)


    def process(self):
        
        tc_splits = self.get_idx_split()["test-challenge"]
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        graph_pos_list = Chem.SDMolSupplier(osp.join(self.original_root, 'pcqm4m-v2-train.sdf'), removeHs=(not self.AddHs))
        smiles_list = data_df['smiles'].tolist() if not self.tc else data_df.iloc[tc_splits]['smiles'].tolist()
        homolumogap_list = data_df['homolumogap'].tolist() if not self.tc else data_df.iloc[tc_splits]['homolumogap'].tolist()
        graph_pos_list = list(graph_pos_list) + [None] * (len(smiles_list) - len(graph_pos_list)) if not self.tc else [None] * len(smiles_list)
        print('Converting SMILES strings and SDF Mol into graphs...')
        data_list = []
        with Pool(processes=120) as pool:
            
            iter = pool.imap(data2graph, zip(smiles_list, graph_pos_list, [self.AddHs] * len(smiles_list)))

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()
                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(graph['position']).to(torch.float32)
                    data_list.append(data)
                    
                except:
                    print('Error in processing graph {}'.format(i))
                    exit()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict
    
    def get_submit_splits(self):
        
        split_dict = self.get_idx_split()
        
        if self.tc:
            split_dict.update({"train": torch.tensor([], dtype=torch.long)})
            split_dict.update({"valid": torch.tensor([], dtype=torch.long)})
            split_dict.update({"test-challenge": torch.arange(len(split_dict["test-challenge"]))})
        
        return split_dict
        

class GlobalPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    
    def __init__(self, root='dataset', smiles2graph=data2graph, AddHs=False, tc=False, transform=None, pre_transform=None):
        super(GlobalPygPCQM4Mv2Dataset, self).__init__(root, smiles2graph, AddHs, tc, transform, pre_transform)
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        return preprocess_item(item)
    
    def download(self):
        return super().download()
    
    def process(self):
        return super().process()
    
class RDKitPCQM4Mv2Dataset(InMemoryDataset):
    
    def __init__(self, root, tc=False, transform=None, pre_transform=None, pre_filter=None):
        
        self.tc = tc
        super(RDKitPCQM4Mv2Dataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        
        return ["whole.pkl"] if not self.tc else ["whole_tc.pkl"]
            
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        
        return ["data.pt"] if not self.tc else ["data_tc.pt"]
    
    def process(self):
        
        if not osp.exists(self.raw_paths[0]):
            # TODO: Run the preprocessing script (bash)
            raise RuntimeError("Please run the preprocessing script first.")
        with open(self.raw_paths[0], "rb") as f:
            rdkit_data = pickle.load(f)
            
        data_list = []
            
        for idx, data in enumerate(tqdm(rdkit_data)):
            
            # atom level properties
            atomic_num = torch.from_numpy(data["atomic_num"]).long()
            chiral_tag = torch.from_numpy(data["chiral_tag"]).long()
            degree = torch.from_numpy(data["degree"]).long()
            explicit_valence = torch.from_numpy(data["explicit_valence"]).long()
            formal_charge = torch.from_numpy(data["formal_charge"]).long()
            hybridization = torch.from_numpy(data["hybridization"]).long()
            implicit_valence = torch.from_numpy(data["implicit_valence"]).long()
            is_aromatic = torch.from_numpy(data["is_aromatic"]).long()
            total_numHs = torch.from_numpy(data["total_numHs"]).long()
            mass = torch.from_numpy(data["mass"]).float()
            
            rdkit_atom_pos = torch.from_numpy(data["rdkit_atom_pos"] - data['rdkit_atom_pos'].mean(axis=0, keepdims=True)).float()

            if data['eq_atom_pos'] is not None:
                eq_atom_pos = torch.from_numpy(data["eq_atom_pos"] - data['eq_atom_pos'].mean(axis=0, keepdims=True)).float()
            else:
                # ! No ground truth atom positions
                eq_atom_pos = torch.zeros_like(rdkit_atom_pos)
            
            # bond level properties
            bond_dir = torch.from_numpy(data["bond_dir"]).long()
            bond_type = torch.from_numpy(data["bond_type"]).long()
            is_in_ring = torch.from_numpy(data["is_in_ring"]).long()
            
            edge_index = torch.from_numpy(data["edges"]).long()
            edge_index = edge_index.T
            
            # graph level properties
            morgan_fp = torch.from_numpy(data["morgan_fp"]).long()
            maccs_fp = torch.from_numpy(data["maccs_fp"]).long()
            daylight_fg_counts = torch.from_numpy(data["daylight_fg_counts"]).long()
            
            y = torch.tensor([data["label"]]).float()
            
            pyg_data = Data(
                atomic_num=atomic_num,
                rdkit_atom_pos=rdkit_atom_pos,
                eq_atom_pos=eq_atom_pos,
                edge_index=edge_index,
                y=y,
                chiral_tag=chiral_tag,
                degree=degree,
                explicit_valence=explicit_valence,
                formal_charge=formal_charge,
                hybridization=hybridization,
                implicit_valence=implicit_valence,
                is_aromatic=is_aromatic,
                total_numHs=total_numHs,
                mass=mass,
                bond_dir=bond_dir,
                bond_type=bond_type,
                is_in_ring=is_in_ring,
                morgan_fp=morgan_fp,
                maccs_fp=maccs_fp,
                daylight_fg_counts=daylight_fg_counts,
                num_nodes=len(atomic_num),
            )
            
            if self.pre_filter is not None and not self.pre_filter(pyg_data):
                continue
            if self.pre_transform is not None:
                pyg_data = self.pre_transform(pyg_data)
            
            data_list.append(pyg_data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])
        
    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict
    
    def get_submit_splits(self):
        
        split_dict = self.get_idx_split()
        
        if self.tc:
            split_dict.update({"train": torch.tensor([], dtype=torch.long)})
            split_dict.update({"valid": torch.tensor([], dtype=torch.long)})
            split_dict.update({"test-challenge": torch.arange(len(split_dict["test-challenge"]))})
        
        return split_dict


if __name__ == '__main__':
    dataset = GlobalPygPCQM4Mv2Dataset(root='data')
    data = dataset[0]
    print(data)
    print('edge_index', data.edge_index)
    print('edge_attr', data.edge_attr)
    print('x', data.x)
    print('pos', data.pos)
    print('attn_bias', data.attn_bias)
    print('attn_edge_type', data.attn_edge_type)
    print('spatial_pos', data.spatial_pos)
    print('in_degree', data.in_degree)
    print('out_degree', data.out_degree)
    print('edge_input', data.edge_input)