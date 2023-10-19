import torch
from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from visnet.models.model import create_model, load_model


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        self._reset_losses_dict()
        self._reset_ema_dict()
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
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        loss_fn = mse_loss if self.hparams.loss_type == 'MSE' else l1_loss
        
        return self.step(batch, loss_fn, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            pred, deriv = self(batch)
        if stage == "test":
            self.inference_results['y_pred'].append(pred.squeeze(-1).detach().cpu())
            self.inference_results['y_true'].append(batch.y.squeeze(-1).detach().cpu())
            if self.hparams.derivative:
                self.inference_results['dy_pred'].append(deriv.squeeze(-1).detach().cpu())
                self.inference_results['dy_true'].append(batch.dy.squeeze(-1).detach().cpu())

        loss_y, loss_dy = 0, 0
        if self.hparams.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0

            loss_dy = loss_fn(deriv, batch.dy)
            
            if stage in ["train", "val"] and self.hparams.loss_scale_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.loss_scale_dy * loss_dy
                    + (1 - self.hparams.loss_scale_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            loss_y = loss_fn(pred, batch.y)
            
            if stage in ["train", "val"] and self.hparams.loss_scale_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.loss_scale_y * loss_y
                    + (1 - self.hparams.loss_scale_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()
            
            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight
        
        self.losses[stage].append(loss.detach())
        
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            delta = 0 if self.hparams.reload == 1 else 1
            should_reset = (
                (self.current_epoch + delta + 1) % self.hparams.test_interval == 0
                or ((self.current_epoch + delta) % self.hparams.test_interval == 0 and self.current_epoch != 0)
            )
            if should_reset:
                self.trainer.reset_val_dataloader()
                self.trainer.fit_loop.epoch_loop.val_loop.epoch_loop._reset_dl_batch_idx(len(self.trainer.val_dataloaders))

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(self.losses["train_dy"]).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

            if len(self.losses["test_y"]) > 0 and len(self.losses["test_dy"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
                result_dict["test_loss_dy"] = torch.stack(self.losses["test_dy"]).mean()

            self.log_dict(result_dict, sync_dist=True)
            
        self._reset_losses_dict()
        self._reset_inference_results()

    def test_epoch_end(self, outputs) -> None:
        for key in self.inference_results.keys():
            if len(self.inference_results[key]) > 0:
                self.inference_results[key] = torch.cat(self.inference_results[key], dim=0)

    def _reset_losses_dict(self):
        self.losses = {
            "train": [], "val": [], "test": [],
            "train_y": [], "val_y": [], "test_y": [],
            "train_dy": [], "val_dy": [], "test_dy": [],
        }

    def _reset_inference_results(self):
        self.inference_results = {'y_pred': [], 'y_true': [], 'dy_pred': [], 'dy_true': []}
        
    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}
