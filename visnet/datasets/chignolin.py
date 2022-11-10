import numpy as np
import torch
from ase.units import Bohr, Hartree
from torch_geometric.data import Data, InMemoryDataset
from tqdm import trange


class Chignolin(InMemoryDataset):
    
    self_energies = {
        1: -0.496665677271,
        6: -37.8289474402,
        7: -54.5677547104,
        8: -75.0321126521,
        16: -398.063946327,
    }

    def __init__(self, root, transform=None, pre_transform=None):
        
        super(Chignolin, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'chignolin.npz']

    @property
    def processed_file_names(self):
        return [f'chignolin.pt']

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            
            data_npz = np.load(path)
            concat_z = torch.from_numpy(data_npz["Z"]).long()
            concat_positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            concat_forces = torch.from_numpy(data_npz["F"]).float() * Hartree / Bohr
            num_atoms = 166

            samples = []
            for index in trange(energies.shape[0]):
                z = concat_z[index * num_atoms:(index + 1) * num_atoms]
                ref_energy = torch.sum(torch.tensor([self.self_energies[int(atom)] for atom in z]))
                pos = concat_positions[index * num_atoms:(index + 1) * num_atoms, :]
                y = (energies[index] - ref_energy) * Hartree
                # ! NOTE: Convert Engrad to Force
                dy = -concat_forces[index * num_atoms:(index + 1) * num_atoms, :]
                data = Data(z=z, pos=pos, y=y.reshape(1, 1), dy=dy)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)