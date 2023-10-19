import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from visnet import datasets, models, priors
from visnet.data import DataModule
from visnet.models import output_modules
from visnet.models.utils import act_class_mapping, rbf_class_mapping
from visnet.module import LNNP
from visnet.utils import LoadFromCheckpoint, LoadFromFile, number, save_argparse


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    
    # training settings
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--loss-type', type=str, default='MSE', choices=['MSE', 'MAE'], help='Loss type')
    parser.add_argument('--loss-scale-y', type=float, default=1.0, help="Scale the loss y of the target")
    parser.add_argument('--loss-scale-dy', type=float, default=1.0, help="Scale the loss dy of the target")
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')
    
    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument')
    parser.add_argument('--dataset-root', default=None, type=str, help='Data storage directory')
    parser.add_argument('--derivative', default=False, action=argparse.BooleanOptionalAction, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--split-mode', default=None, type=str, help='Split mode for Molecule3D dataset')
    
    # dataloader specific
    parser.add_argument('--reload', type=int, default=0, help='Reload dataloaders every n epoch')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=950, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=50, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=None, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    
    # model architecture specific
    parser.add_argument('--model', type=str, default='ViSNetBlock', choices=models.__all__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Which prior model to use')
    parser.add_argument('--prior-args', type=dict, default=None, help='Additional arguments for the prior model')
    
    # architectural specific
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', action=argparse.BooleanOptionalAction, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff in model')
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')
    parser.add_argument('--lmax', type=int, default=2, help='Max order of spherical harmonics')
    parser.add_argument('--vecnorm-type', type=str, default='max_min', help='Type of vector normalization')
    parser.add_argument('--trainable-vecnorm', action=argparse.BooleanOptionalAction, default=False, help='If vector normalization should be trainable')
    parser.add_argument('--vertex-type', type=str, default='Edge', choices=['None', 'Edge', 'Node'], help='If add vertex angle and Where to add vertex angles')

    # other specific
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', type=str, default=None, help='Log directory')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'inference'], help='Train or inference') 
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend')
    parser.add_argument('--redirect', action=argparse.BooleanOptionalAction, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--accelerator', default='gpu', help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    
    args = parser.parse_args()

    if args.redirect:
        os.makedirs(args.log_dir, exist_ok=True)
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(logging.StreamHandler(sys.stdout))

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size
    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])
    
    return args

def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    data = DataModule(args)
    data.prepare_dataset()

    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    dir_name = f"output_ngpus_{len(cuda_visible_devices)}_bs_{args.batch_size}_lr_{args.lr}_seed_{args.seed}" + \
               f"_reload_{args.reload}_lmax_{args.lmax}_vnorm_{args.vecnorm_type}" + \
               f"_vertex_{args.vertex_type}_L{args.num_layers}_D{args.embedding_dimension}_H{args.num_heads}" + \
               f"_cutoff_{args.cutoff}_E{args.energy_weight}_F{args.force_weight}_loss_{args.loss_type}"
    
    if args.load_model is None:
        args.log_dir = os.path.join(args.log_dir, dir_name)
        if os.path.exists(args.log_dir):
            if os.path.exists(os.path.join(args.log_dir, "last.ckpt")):
                args.load_model = os.path.join(args.log_dir, "last.ckpt")
            csv_path = os.path.join(args.log_dir, "metrics.csv")
            while os.path.exists(csv_path):
                csv_path = csv_path + '.bak'
            if os.path.exists(os.path.join(args.log_dir, "metrics.csv")):
                os.rename(os.path.join(args.log_dir, "metrics.csv"), csv_path)

    prior = None
    if args.prior_model:
        assert hasattr(priors, args.prior_model), (
            f"Unknown prior model {args['prior_model']}. "
            f"Available models are {', '.join(priors.__all__)}"
        )
        # initialize the prior model
        prior = getattr(priors, args.prior_model)(dataset=data.dataset)
        args.prior_args = prior.get_init_args()

    # initialize lightning module
    model = LNNP(args, prior_model=prior, mean=data.mean, std=data.std)

    if args.task == "train":
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.log_dir,
            monitor="val_loss",
            save_top_k=10,
            save_last=True,
            every_n_epochs=args.save_interval,
            filename="{epoch}-{val_loss:.4f}-{test_loss:.4f}",
        )
        
        early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)
        tb_logger = TensorBoardLogger(args.log_dir, name="tensorbord", version="", default_hp_metric=False)
        csv_logger = CSVLogger(args.log_dir, name="", version="")
        ddp_plugin = DDPStrategy(find_unused_parameters=False)

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            gpus=args.ngpus,
            num_nodes=args.num_nodes,
            accelerator=args.accelerator,
            default_root_dir=args.log_dir,
            auto_lr_find=False,
            callbacks=[early_stopping, checkpoint_callback],
            logger=[tb_logger, csv_logger],
            reload_dataloaders_every_n_epochs=args.reload,
            precision=args.precision,
            strategy=ddp_plugin,
            enable_progress_bar=True,
        )

        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)

    test_trainer = pl.Trainer(
        logger=False,
        max_epochs=-1,
        num_nodes=1,
        gpus=1,
        default_root_dir=args.log_dir,
        enable_progress_bar=True,
        inference_mode=False,
    )
        
    if args.task == 'train':
        test_trainer.test(model=model, ckpt_path=trainer.checkpoint_callback.best_model_path, datamodule=data)
    elif args.task == 'inference':
        test_trainer.test(model=model, datamodule=data)
        torch.save(model.inference_results, os.path.join(args.log_dir, "inference_results.pt"))
    
    emae = np.abs(model.inference_results['y_true'].numpy() - model.inference_results['y_pred'].numpy()).mean()
    print('Scalar MAE: {:.6f}'.format(emae))
    if args.derivative:
        fmae = np.abs(model.inference_results['dy_true'].numpy() - model.inference_results['dy_pred'].numpy()).mean()
        print('Forces MAE: {:.6f}'.format(fmae))

if __name__ == "__main__":
    main()
