import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from pytorch_lightning import LightningModule
from OGB_ViSNet.Pretrained_3D_ViSNet.model import create_model, load_model, create_clip_model, load_clip_model
from OGB_ViSNet.utils import InfoNCELoss

loss_mapping_class = dict(l1=l1_loss, mse=mse_loss, InfoNCE=InfoNCELoss)

class LNNP_Pretrained_3D_ViSNet(LightningModule):
    def __init__(self, hparams, mean=None, std=None):
        super(LNNP_Pretrained_3D_ViSNet, self).__init__()

        self.save_hyperparameters(hparams)
        
        if self.hparams.load_teacher_model and not self.hparams.load_model:
            print("Loading teacher model...")
            teacher_model = load_model(self.hparams.load_teacher_model)
            print("Freezing teacher model...")
            for param in teacher_model.parameters():
                param.requires_grad = False

        if self.hparams.load_model:
            self.model = load_clip_model(student_filepath=self.hparams.load_model, teacher_filepath=self.hparams.load_teacher_model)
        else:
            student_model = create_model(self.hparams, mean, std)
            self.model = create_clip_model(student_model=student_model, teacher_model=teacher_model)

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()
        self._reset_inference_results()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_epoch_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, data, stage):
        return self.model(data, stage)

    def training_step(self, batch, batch_idx):
        loss_fn = loss_mapping_class[self.hparams.loss_type]
        return self.step(batch, loss_fn, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
  
        with torch.set_grad_enabled(stage == "train"):
            out_eq, out_rdkit, pred_rdkit, verify_eq = self(batch, stage)

        if stage == 'test' and pred_rdkit is not None:
            self.inference_results['y_pred'].append(pred_rdkit.squeeze(-1))
            self.inference_results['y_true'].append(batch['y'].squeeze(-1))
            return None
            
        loss = 0
        loss_h = loss_e = nll_acc = loss_rdkit = loss_eq = torch.tensor(0.0, requires_grad=True) # consist with loss.detach()

        if batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)
        
        if stage == "train":
            loss_h = l1_loss(out_eq, out_rdkit)
            loss_e = loss_fn(pred_rdkit, batch.y)
            loss = self.hparams.loss_e_weight * loss_e + self.hparams.loss_h_weight * loss_h
            loss_eq = loss_fn(verify_eq, batch.y)
        else:
            loss_e = loss_fn(pred_rdkit, batch.y)
            loss = loss_e
            
        self.losses[stage].append(loss.detach())
        
        self.losses[f"{stage}_top1_acc"].append(nll_acc.detach())
        self.losses[f"{stage}_h"].append(loss_h.detach())
        self.losses[f"{stage}_e"].append(loss_e.detach())
        self.losses[f"{stage}_rdkit"].append(loss_rdkit.detach())
        self.losses[f"{stage}_eq"].append(loss_eq.detach())
        
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_epoch_loss": torch.stack(self.losses["train"]).mean(),
                "train_epoch_e": torch.stack(self.losses["train_e"]).mean(),
                "train_epoch_h": torch.stack(self.losses["train_h"]).mean(),
                "train_epoch_rdkit": torch.stack(self.losses["train_rdkit"]).mean(),
                "train_epoch_eq": torch.stack(self.losses["train_eq"]).mean(),
                "val_epoch_loss": torch.stack(self.losses["val"]).mean(),
                "val_epoch_e": torch.stack(self.losses["val_e"]).mean(),
                "val_epoch_h": torch.stack(self.losses["val_h"]).mean(),
                "val_epoch_top1_acc": torch.stack(self.losses["val_top1_acc"]).mean(),
                "val_epoch_rdkit": torch.stack(self.losses["val_rdkit"]).mean(),
            }

            self.log_dict(result_dict, sync_dist=True)
            
        self._reset_losses_dict()
        self.results = []

    def test_epoch_end(self, outputs) -> None:
        for key in self.inference_results.keys():
            self.inference_results[key] = torch.cat(self.inference_results[key], dim=0)

    def _reset_losses_dict(self):

        self.losses = {"train": [], "val": []}
        self.losses.update({"train_h": [], "train_top1_acc": [], "val_h": [], "val_top1_acc": []})
        self.losses.update({"train_e": [], "val_e": []})
        self.losses.update({"train_rdkit": [], "val_rdkit": []})
        self.losses.update({"train_eq": [], "val_eq": []})

    def _reset_inference_results(self):
        self.inference_results = {'y_pred': [], 'y_true': []}