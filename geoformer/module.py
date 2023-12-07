from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from geoformer.model.modeling_geoformer import create_model


class LNNP(LightningModule):
    def __init__(self, config) -> None:
        super(LNNP, self).__init__()

        self.save_hyperparameters(config)
        self.model = create_model(self.hparams)
        self._reset_losses_dict()

    def configure_optimizers(self) -> Optional[AdamW]:
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_cosine_length,
                eta_min=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise NotImplementedError(
                f"Unknown lr_schedule: {self.hparams.lr_scheduler}"
            )

        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        return self.model(z=batch["z"], pos=batch["pos"])

    def training_step(self, batch, batch_idx):
        if self.hparams.loss_type == "MSE":
            return self.step(batch, mse_loss, "train")
        elif self.hparams.loss_type == "MAE":
            return self.step(batch, l1_loss, "train")
        else:
            NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)

        loss = 0

        if "labels" in batch:
            if batch["labels"].ndim == 1:
                batch["labels"] = batch["labels"].unsqueeze(1)

            loss = loss_fn(pred, batch["labels"])
            self.losses[stage].append(loss.detach())

        self.losses[stage].append(loss.detach())

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

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(
                    self.losses["test"]
                ).mean()

            self.log_dict(result_dict, prog_bar=True, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        result_dict = {}
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
        self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }
