import yaml
import argparse
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
import torch.nn.functional as F

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from os.path import dirname
import pytorch_lightning as pl
from pytorch_lightning.plugins import ApexMixedPrecisionPlugin
from pytorch_lightning.utilities import _APEX_AVAILABLE
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
if _APEX_AVAILABLE:
    from apex import amp

def InfoNCELoss(rep_x, rep_y, **kwargs):
    
    tau = kwargs["tau"]
    rep_x = F.normalize(rep_x, dim=-1)
    rep_y = F.normalize(rep_y, dim=-1)
    criterion = nn.CrossEntropyLoss()
    B = rep_x.shape[0]
    logits = torch.mm(rep_x, rep_y.T) / tau
    labels = torch.arange(B, device=logits.device, dtype=torch.long)
    nll = criterion(logits, labels)
    pred = logits.argmax(dim=-1)
    nll_acc = pred.eq(labels).sum().detach() * 1. / B
    
    return nll, nll_acc

class OGBTransform(BaseTransform):
    
    def __init__(self, distance_otf: str, cutoff: float, max_num_neighbors: int) -> None:
        super().__init__()
        
        self.distance_otf = distance_otf
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
    
    def __call__(self, data: Data) -> Data:
        
        if self.distance_otf:
            if data.eq_atom_pos is not None:
                data.eq_edge_index = radius_graph(data.eq_atom_pos, r=self.cutoff, batch=data.batch, loop=True, max_num_neighbors=self.max_num_neighbors)
            data.rdkit_edge_index = radius_graph(data.rdkit_atom_pos, r=self.cutoff, batch=data.batch, loop=True, max_num_neighbors=self.max_num_neighbors)
            
        else:
            data.eq_edge_index = data.edge_index
            data.rdkit_edge_index = data.edge_index
        return data
    
class CustomApexMixedPrecisionPlugin(ApexMixedPrecisionPlugin):
    def __init__(self, amp_level, max_loss_scale=128, min_loss_scale=0.0001, **kwargs):
        super().__init__(amp_level=amp_level, **kwargs)
        self.max_loss_scale = max_loss_scale
        self.min_loss_scale = min_loss_scale

    def dispatch(self, trainer: "pl.Trainer") -> None:
        if not self._connected:
            accelerator = trainer.accelerator
            _, accelerator.optimizers = amp.initialize(
                trainer.lightning_module, accelerator.optimizers, opt_level=self.amp_level, 
                max_loss_scale=self.max_loss_scale, min_loss_scale=self.min_loss_scale
            )
            self._connected = True
        return MixedPrecisionPlugin().dispatch(trainer)


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        ckpt = torch.load(values, map_location="cpu")
        config = ckpt["hyper_parameters"]
        for key in config.keys():
            if key not in namespace:
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    import os
    os.makedirs(dirname(filename), exist_ok=True)
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


class MissingLabelException(Exception):
    pass

class PolynomialDecayLRSchedule(_LRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, total_num_updates, warmup_updates=0, end_learning_rate=1e-9, power=1.0, last_epoch=-1):

        assert total_num_updates > 0

        self.lrs = [group["lr"] for group in optimizer.param_groups]
    
        self.warmup_updates = warmup_updates
        self.end_learning_rate = end_learning_rate
        self.total_num_updates = total_num_updates
        self.power = power
        
        super(PolynomialDecayLRSchedule, self).__init__(optimizer, last_epoch)
        

    def get_lr(self):
        """Update the learning rate after each update."""
        num_updates = self.last_epoch
        
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            self.warmup_factor = num_updates / float(self.warmup_updates)
            lr = list(map(lambda init_lr: self.warmup_factor * init_lr, self.lrs))
        elif num_updates >= self.total_num_updates:
            lr = [self.end_learning_rate for _ in range(len(self.lrs))]
        else:
            warmup = self.warmup_updates
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_updates - warmup)
            lr = list(map(lambda init_lr: (init_lr - self.end_learning_rate) * pct_remaining ** (self.power) + self.end_learning_rate, self.lrs))
        return lr