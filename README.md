# Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing

## Overview

ViSNet (shorted for “**V**ector-**S**calar **i**nteractive graph neural **Net**work”) is an equivariant geometry-enhanced graph neural for molecules that significantly alleviate the dilemma between computational costs and sufficient utilization of geometric information.

<img src="visnet_arch.png" width=100%> 

## News

### Aug 2023

- *ViSNet-Drug Team* won the 1st place in the [The First Global AI Drug Development Competition](https://aistudio.baidu.com/competition/detail/1012/0/leaderboard)!

### Nov 2022
- *ViSNet Team* won the 2nd place in the [OGB-LSC @ NeurIPS 2022 PCQM4Mv2 Track](https://ogb.stanford.edu/neurips2022/results/)! Please check out the branch [OGB-LSC@NIPS2022](https://github.com/microsoft/ViSNet/tree/OGB-LSC%40NIPS2022) and give it a star if you find it useful!

## Environments

- Clone this repository

- Install the dependencies

```shell
conda create -y -n visnet python=3.9
conda activate visnet
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg==2.1.0 -c pyg
pip install pytorch-lightning==1.8.0
pip install ase ase[test] ogb
```

## Getting started

To train ViSNet on MD17, just run:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --conf examples/ViSNet-MD17.yml --dataset-arg aspirin --dataset-root /path/to/data --log-dir /path/to/log
```

One can modify the ```dataset-arg``` to train another molecule like ethanol.

To train ViSNet on QM9, just run:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --conf examples/ViSNet-QM9.yml --dataset-arg energy_U0 --dataset-root /path/to/data --log-dir /path/to/log
```

One can modify the ```dataset-arg``` to train another property like energy_U.

We have also provided example training configuration files for other datasets within the [examples](./examples/), which can be used in a similar way.

## Inference

Once ViSNet is trained, to use a pretrained checkpoint for inference, simply run:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --conf examples/ViSNet-MD17.yml --dataset-arg aspirin --dataset-root /path/to/data --log-dir /path/to/log --task inference --load-model /path/to/ckpt
```

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --conf examples/ViSNet-QM9.yml --dataset-arg energy_U0 --dataset-root /path/to/data --log-dir /path/to/log --task inference --load-model /path/to/ckpt
```

## Contact

Please contact Dr. Tong Wang (watong@microsoft.com) for technical support.

## License

This project is licensed under the terms of the MIT license. 
