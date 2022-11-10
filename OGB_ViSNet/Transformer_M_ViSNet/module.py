import torch
import torch.nn as nn
from torch.optim import AdamW
from OGB_ViSNet.utils import PolynomialDecayLRSchedule

from pytorch_lightning import LightningModule
from OGB_ViSNet.Transformer_M_ViSNet.model import create_model, load_model

class LNNP_Transformer_M_ViSNet(LightningModule):
    def __init__(self, hparams, mean=None, std=None):
        super(LNNP_Transformer_M_ViSNet, self).__init__()
        
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model)
        else:
            self.model = create_model(self.hparams, mean, std)

        # initialize loss collection
        self._reset_losses_dict()
        self._reset_inference_results()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = PolynomialDecayLRSchedule(
            optimizer,
            self.hparams.total_num_updates,
            self.hparams.warmup_updates,
            self.hparams.end_learning_rate,
            self.hparams.power,
        )    
        lr_scheduler = {
            'scheduler': scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]  

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def step(self, batch, stage):
        
        sample_size = batch['pos'].shape[0]
        
        with torch.set_grad_enabled(stage == "train"):
            ori_pos = batch['pos']
            noise = torch.randn(ori_pos.shape).to(ori_pos) * self.hparams.noise_scale
            noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
            noise = noise.masked_fill_(noise_mask, 0.0)
            batch['pos'] = ori_pos + noise
            model_output = self(batch)
            logits, node_output = model_output[0], model_output[1]
            logits = logits[:, 0, :]
            
            if stage == "test":
                self.inference_results['y_pred'].append(logits.squeeze(-1))
                self.inference_results['y_true'].append(batch['y'].squeeze(-1))
                return None
            
            loss = nn.L1Loss(reduction='sum')(logits.squeeze(-1), batch['y'])
            if node_output is not None:
                node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + noise_mask
                node_output = node_output.masked_fill_(node_mask, 0.0)

                node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
                node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)

                tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
                tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
                node_output_loss = (node_output_loss / tgt_count).sum()
            else:
                node_output_loss = (noise - noise).sum()

        self.logging_info[f'{stage}_loss'].append(loss.detach())
        self.logging_info[f'{stage}_node_output_loss'].append(node_output_loss.detach())
        self.logging_info[f'{stage}_sample_size'].append(sample_size)
        
        if stage == 'train':
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{stage}_step_loss', loss / sample_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{stage}_step_node_output_loss', node_output_loss / sample_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
        return (loss + node_output_loss) / sample_size

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:   
            train_sample_size = torch.sum(torch.tensor(self.logging_info["train_sample_size"]))
            valid_sample_size = torch.sum(torch.tensor(self.logging_info["val_sample_size"]))
            
            result_dict = {
                "epoch": float(self.current_epoch),
                "train_epoch_loss": torch.stack(self.logging_info["train_loss"]).sum() / train_sample_size,
                "train_epoch_node_output_loss": torch.stack(self.logging_info["train_node_output_loss"]).sum() / train_sample_size,
                "val_epoch_loss": torch.stack(self.logging_info["val_loss"]).sum() / valid_sample_size,
                "val_epoch_node_output_loss": torch.stack(self.logging_info["val_node_output_loss"]).sum() / valid_sample_size,
            }

            self.log_dict(result_dict, logger=True, sync_dist=True)
            
        self._reset_losses_dict()
    
    def test_epoch_end(self, outputs) -> None:
        for key in self.inference_results.keys():
            self.inference_results[key] = torch.cat(self.inference_results[key], dim=0)

    def _reset_losses_dict(self):
        self.logging_info = {
            "train_loss": [], 
            "val_loss": [], 
            'train_node_output_loss':[], 
            'val_node_output_loss': [],
            "train_sample_size": [], 
            "val_sample_size": [],
        }
        
    def _reset_inference_results(self):
        self.inference_results = {'y_pred': [], 'y_true': []}

