from tqdm import tqdm
import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn

from OGB_ViSNet.utils import MissingLabelException
from pytorch_lightning.utilities import rank_zero_only
from OGB_ViSNet.datasets.utils import collator
from OGB_ViSNet.datasets import GlobalPygPCQM4Mv2Dataset, RDKitPCQM4Mv2Dataset
from OGB_ViSNet.utils import OGBTransform
from functools import partial

class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        
        self.dataset = dataset
        if self.dataset is None:
            if self.hparams["is_submit"]:
                assert self.hparams["inference_dataset"] == "test-challenge"
            if self.hparams["model_choice"] == "Transformer_M_ViSNet":
                self.dataset = GlobalPygPCQM4Mv2Dataset(root=self.hparams["dataset_root"], AddHs=self.hparams['AddHs'], tc=self.hparams["is_submit"])
            elif self.hparams["model_choice"] == "Pretrained_3D_ViSNet":
                transform = OGBTransform(self.hparams.distance_otf, self.hparams.cutoff_upper, self.hparams.max_num_neighbors)
                self.dataset = RDKitPCQM4Mv2Dataset(root=self.hparams["dataset_root"], transform=transform, tc=self.hparams["is_submit"])

    def split_compute(self):
            
        split_idx = self.dataset.get_submit_splits()
        
        print(f"train {len(split_idx['train'])}, val {len(split_idx['valid'])}, test {len(split_idx[self.hparams.get('inference_dataset', 'valid')])}")
        print(f"length of dataset: {len(self.dataset)}")
        
        self.train_dataset = self.dataset.index_select(split_idx["train"][:5000])
        self.val_dataset = self.dataset.index_select(split_idx["valid"])
        self.test_dataset = self.dataset.index_select(split_idx[self.hparams.get("inference_dataset", "valid")])

        if self.hparams["standardize"] and self.hparams['task'] == 'train':
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

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
            
        dl_params = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )
        
        if self.hparams["model_choice"] == "Transformer_M_ViSNet":
            from torch.utils.data import DataLoader
            dl = DataLoader(**dl_params, collate_fn=partial(collator, max_node=256, multi_hop_max_dist=5, spatial_pos_max=1024))
        elif self.hparams["model_choice"] == "Pretrained_3D_ViSNet":
            from torch_geometric.loader import DataLoader
            dl = DataLoader(**dl_params)

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        
        def get_label(batch):
            if batch['y'] is None:
                raise MissingLabelException()
            return batch['y'].squeeze().clone()

        data = tqdm(self._get_dataloader(self.train_dataset, "val", store_dataloader=False), desc="computing mean and std",)
        
        try:
            ys = torch.cat([get_label(batch) for batch in data])
        except MissingLabelException:
            rank_zero_warn("Standardize is true but failed to compute dataset mean and standard deviation.")
            return None

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
