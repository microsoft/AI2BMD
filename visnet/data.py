from os.path import join

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

from visnet.datasets import *
from visnet.utils import MissingLabelException, make_splits


class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super(DataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = None

    def prepare_dataset(self):
        
        assert hasattr(self, f"_prepare_{self.hparams['dataset']}_dataset"), f"Dataset {self.hparams['dataset']} not defined"
        dataset_factory = lambda t: getattr(self, f"_prepare_{t}_dataset")()
        self.idx_train, self.idx_val, self.idx_test = dataset_factory(self.hparams["dataset"])
            
        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        delta = 1 if self.hparams['reload'] == 1 else 2
        if (
            len(self.test_dataset) > 0
            and (self.trainer.current_epoch + delta) % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        def get_label(batch, atomref):
            if batch.y is None:
                raise MissingLabelException()

            if atomref is None:
                return batch.y.clone()

            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False), 
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            ys = torch.cat([get_label(batch, atomref) for batch in data])
        except MissingLabelException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return None

        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
    
    def _prepare_Chignolin_dataset(self):
        
        self.dataset = Chignolin(root=self.hparams["dataset_root"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        
        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test
    
    def _prepare_MD17_dataset(self):
        
        self.dataset = MD17(root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        
        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test

    def _prepare_MD22_dataset(self):
        
        self.dataset = MD22(root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
        train_val_size = self.dataset.molecule_splits[self.hparams["dataset_arg"]]
        train_size = round(train_val_size * 0.95)
        val_size = train_val_size - train_size
        
        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test

    def _prepare_Molecule3D_dataset(self):
        
        self.dataset = Molecule3D(root=self.hparams["dataset_root"])
        split_dict = self.dataset.get_idx_split(self.hparams['split_mode'])
        idx_train = split_dict['train']
        idx_val = split_dict['valid']
        idx_test = split_dict['test']
        
        return idx_train, idx_val, idx_test
    
    def _prepare_QM9_dataset(self):
        
        self.dataset = QM9(root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        
        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test
    
    def _prepare_rMD17_dataset(self):
        
        self.dataset = rMD17(root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        
        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test
    