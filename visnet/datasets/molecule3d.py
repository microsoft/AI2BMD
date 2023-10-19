import json
import os.path as osp
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class Molecule3D(InMemoryDataset):
    
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        
        self.root = root
        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'molecule3d.pt'
    
    def process(self):
        
        data_list = []
        sdf_paths = [
            osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
            osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
            osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
            osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')
        ]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]
        
        
        target_path = osp.join(self.raw_dir, 'properties.csv')
        target_df = pd.read_csv(target_path)
        
        abs_idx = -1
        
        for i, suppl in enumerate(suppl_list):
            with Pool(processes=120) as pool:
                iter = pool.imap(self.mol2graph, suppl)
                for j, graph in tqdm(enumerate(iter), total=len(suppl)):
                    abs_idx += 1
                    
                    data = Data()
                    data.__num_nodes__ = int(graph['num_nodes'])
                    
                    # Required by GNNs
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.FloatTensor([target_df.iloc[abs_idx, 6]]).unsqueeze(1)
                    
                    # Required by ViSNet
                    data.pos = torch.tensor(graph['position'], dtype=torch.float32)
                    data.z = torch.tensor(graph['z'], dtype=torch.int64)
                    data_list.append(data)
                    
        torch.save(self.collate(data_list), self.processed_paths[0])
    
    def get_idx_split(self, split_mode='random'):
        assert split_mode in ['random', 'scaffold']
        split_dict = json.load(open(osp.join(self.raw_dir, f'{split_mode}_split_inds.json'), 'r'))
        for key, values in split_dict.items():
            split_dict[key] = torch.tensor(values)
        return split_dict
                  
    def mol2graph(self, mol):
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)
        
        coords = mol.GetConformer().GetPositions()
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['position'] = coords
        graph['z'] = z

        return graph 
