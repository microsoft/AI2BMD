import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


class MD22(InMemoryDataset):
    def __init__(self, root, dataset_arg=None, transform=None, pre_transform=None):
        
        self.dataset_arg = dataset_arg
        
        super(MD22, self).__init__(osp.join(root, dataset_arg), transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def molecule_names(self):
        
        molecule_names = dict(
            Ac_Ala3_NHMe="md22_Ac-Ala3-NHMe.npz",
            DHA="md22_DHA.npz",
            stachyose="md22_stachyose.npz",
            AT_AT="md22_AT-AT.npz",
            AT_AT_CG_CG="md22_AT-AT-CG-CG.npz",
            buckyball_catcher="md22_buckyball-catcher.npz",
            double_walled_nanotube="md22_dw_nanotube.npz"
        )

        return molecule_names

    @property
    def raw_file_names(self):
        return [self.molecule_names[self.dataset_arg]]

    @property
    def processed_file_names(self):
        return [f"md22_{self.dataset_arg}.pt"]
    
    @property
    def base_url(self):
        return "http://www.quantum-machine.org/gdml/data/npz/"

    def download(self):
        
        download_url(self.base_url + self.molecule_names[self.dataset_arg], self.raw_dir)
            
    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()

            samples = []
            for pos, y, dy in tqdm(zip(positions, energies, forces), total=energies.size(0)):
                
                data = Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)
    
    @property
    def molecule_splits(self):
        """
            Splits refer to MD22 https://arxiv.org/pdf/2209.14865.pdf
        """
        return dict(
            Ac_Ala3_NHMe=6000,
            DHA=8000,
            stachyose=8000,
            AT_AT=3000,
            AT_AT_CG_CG=2000,
            buckyball_catcher=600,
            double_walled_nanotube=800
        )