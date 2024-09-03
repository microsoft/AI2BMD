# AI<sup>2</sup>BMD: AI-powered *ab initio* biomolecular dynamics simulation

## Contents

- [Overview](#overview)
- [Setup Guide](#ai2bmd-setup-guide)
- [Datasets](#datasets)
- [System Requirements](#system-requirements)
- [Related Research](#ai2bmd-related-research)
- [Citation](#citation)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Contacts](#contacts)

## Overview

AI<sup>2</sup>BMD is a program for efficiently simulating protein molecular dynamics with *ab initio* accuracy. This repository contains simulation programs, datasets, and public materials related to AI<sup>2</sup>BMD.

<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/ai2bmd_logo.png?raw=true" width=50%>

## AI<sup>2</sup>BMD Setup Guide

The source code of AI<sup>2</sup>BMD is hosted in this repository. 
We package the source code and runtime libraries into a Docker image, and provide a Python launcher program to simplify the setup process.
To run the simulation program, you don't need to clone this repository. Simply download `scripts/ai2bmd` and launch it (Python >=3.7 is required).

```shell
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/scripts/ai2bmd'
chmod +x ai2bmd
# you may need to "sudo" the following line if the docker group is not configured for the user
./ai2bmd --prot-file path/to/target-protein.pdb --sim-steps nnn  ...
#        '-------- required argument ---------' '-- optional arguments --'
#
# Notable optional arguments:
#
# [Simulation directory mapping options]
#   --base-dir path/to/base-dir    A directory for running simulation (defaults to current directory)
#   --log-dir  path/to/log-dir     A directory for saving results (defaults to base-dir/Logs-protein-name)
#
# [Simulation parameter options]
#   --sim-steps nnn                Simulation steps
#   --temp-k nnn                   Simulation temperature in Kelvin
#   --timestep nnn                 TimeStep (fs) for simulation
#   --preeq-steps nnn              Pre-equilibration simulation steps for each constraint
#   --max-cyc nnn                  Maximum energy minimization cycles in preprocessing
#
# [Performance tweaks]
#   --device-strategy [strategy]   The compute device allocation strategy
#       small-molecule             Bonded/non-bonded/solvent computation share all GPUs, enable GPU oversubscription
#       large-molecule             No multiple models on the same GPU
#   --chunk-size nnn               When there's more than device_chunk elements (e.g. dipeptides) in a batch, split them into chunks
#                                  and feed them into GPUs sequentially. Reduces memory consumption
#
# [Additional launcher options]
#   --software-update              When specified, updates the program in the Docker image before running
#   --download-training-data       When specified, downloads the AI2BMD training data, and unpacks it in the working directory. 
#                                  Ignores all other options.
#   --gpus                         Specifies the GPU devices to passthrough to the program. Can be one of the following:
#                                  all:        Passthrough all available GPUs to the program.
#                                  none:       Disables GPU passthrough.
#                                  i[,j,k...]  Passthrough some GPUs. Example: --gpus 0,1
```

### Running Simulation

We can run a molecular dynamics simulation as follows.

```shell
# skip the following two lines if you've already set up the launcher
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/scripts/ai2bmd'
chmod +x ai2bmd
# download the Chignolin protein structure data file
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/resources/samples/chig.pdb'
# launch the program, with all simulation parameters set to default values
# you may need to "sudo" the following line if the docker group is not configured for the user
./ai2bmd --prot-file chig.pdb
```

Here we use a very simple protein `Chignolin` as an example.
The program will run a simulation with the default parameters.

The results will be placed in a new directory `Logs-chig`.
The directory contains the simulation trajectory file:

- chig-traj.traj: The full trajectory file in ASE binary format.

## Datasets

### Protein Unit Dataset

The protein unit dataset covers a wide range of conformations for dipeptides. It can be downloaded with the following commands: 

```shell
# skip the following two lines if you've already set up the launcher
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/scripts/ai2bmd'
chmod +x ai2bmd
# you may need to "sudo" the following line if the docker group is not configured for the user
./ai2bmd --download-training-data
```

When it finishes, the current working directory will be populated by the numpy data files (*.npz).

### AIMD-Chig Dataset

The AIMD-Chig consists of 2M conformations of the 166-atom *Chignolin* and their corresponding potential energy and atomic forces calculated using DFT (M06-2X/6-31g*) level.

<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/aimd-chig.png?raw=true" width=50%>-->

- Read the article [AIMD-Chig: Exploring the conformational space of a 166-atom protein Chignolin with ab initio molecular dynamics](https://www.nature.com/articles/s41597-023-02465-9).

- Find the story [The first whole conformational molecular dynamics dataset for proteins at ab initio accuracy and the novel computational technologies behind it](https://bioengineeringcommunity.nature.com/posts/aimd-chig-exploring-the-conformational-space-of-proteins-at-dft-level).

- Get the dataset [AIMD-Chig](https://figshare.com/articles/dataset/_strong_AIMD-Chig_exploring_the_conformational_space_of_166-atom_protein_strong_em_strong_Chignolin_strong_em_strong_with_strong_em_strong_ab_initio_strong_em_strong_molecular_dynamics_strong_/22786730).

## System Requirements

### Hardware Requirements

The AI<sup>2</sup>BMD program runs on x86-64 GNU/Linux systems.
We recommend a machine with the following specs:

- **CPU**: 8+ cores
- **Memory**: 32+ GB
- **GPU**: CUDA-enabled GPU with 8+ GB memory

The program has been tested on the following GPUs:
- A100
- V100
- RTX A6000
- Titan RTX

### Software Requirements

The program has been tested on the following systems:

- **OS**: Ubuntu 20.04,  **Docker**: 27.1
- **OS**: ArchLinux,  **Docker**: 26.1

##  AI<sup>2</sup>BMD Related Research

### Model Architectures

#### ViSNet

ViSNet (**V**ector-**S**calar **i**nteractive graph neural **Net**work) is an equivariant geometry-enhanced graph neural for molecules that significantly alleviates the dilemma between computational costs and the sufficient utilization of geometric information.

<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/visnet_arch.png?raw=true" width=50%>-->

- ViSNet is published on *Nature Communications* [Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing](https://www.nature.com/articles/s41467-023-43720-2).

- ViSNet is selected as "Editors' Highlights" for both ["**AI and machine learning**"](https://www.nature.com/collections/ceiajcdbeb) and ["**Biotechnology and methods**"](https://www.nature.com/collections/idhhgedgig) fields of Nature Communications.

<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/ai-eh.png?raw=true" width=50%>-->
<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/bio-eh.png?raw=true" width=50%> -->

- ViSNet has won the Championship in [The First Global AI Drug Development Competition](https://aistudio.baidu.com/competition/detail/1012/0/leaderboard) and one of the winners in [OGB-LSC @ NeurIPS 2022 PCQM4Mv2 Track](https://ogb.stanford.edu/neurips2022/results/)! 

- Please check out the branch [ViSNet](https://github.com/microsoft/AI2BMD/tree/ViSNet) for the source code, instructions on model training, and more techniqucal details.
  
#### Geoformer

Geoformer (**Geo**metric Trans**former**) is a novel geometric Transformer to effectively model molecular structures for various molecular property predictions. Geoformer introduces a novel positional encoding method, Interatomic Positional Encoding (IPE), to parameterize atomic environments in Transformer. By incorporating IPE, Geoformer captures valuable geometric information beyond pairwise distances within a Transformer-based architecture. Geoformer can be regarded as a Transformer variant of ViSNet.

- Geoformer was published on NeurIPS 2023.
- Read the paper of Geoformer [Geometric Transformer with Interatomic Positional Encoding](https://github.com/microsoft/AI2BMD/tree/Geoformer/Geoformer.pdf).
- Please check out the branch [Geoformer](https://github.com/microsoft/AI2BMD/tree/Geoformer) for the source code, instructions on model training, and more techniqucal details.

<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/geoformer.png?raw=true" width=50%>-->

#### Fine-grained force metrics for MLFF

Machine learning force fields (MLFFs) have gained popularity in recent years as a cost-effective alternative to *ab initio* molecular dynamics (MD) simulations. Despite their small errors on test sets, MLFFs inherently suffer from generalization and robustness issues during MD simulations.

To alleviate these issues, we propose the use of global force metrics and fine-grained metrics from elemental and conformational aspects to systematically measure MLFFs for every atom and conformation of molecules. Furthermore, the performance of MLFFs and the stability of MD simulations can be enhanced by employing the proposed force metrics during model training. This includes training MLFF models using these force metrics as loss functions, fine-tuning by reweighting samples in the original dataset, and continued training by incorporating additional unexplored data.

<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/mlff.jpg?raw=true" width=25%>-->

- Read the Cover Story article [Improving machine learning force fields for molecular dynamics simulations with fine-grained force metrics](https://pubs.aip.org/aip/jcp/article-abstract/159/3/035101/2902663/Improving-machine-learning-force-fields-for?redirectedFrom=fulltext) .

#### Stochastic lag time parameterization for Markov State Model

Markov state models (MSMs) play a key role in studying protein conformational dynamics. A sliding count window with a fixed lag time is commonly used to sample sub-trajectories for transition counting and MSM construction. However, sub-trajectories sampled with a fixed lag time may not perform well under different selections of lag time, requiring strong prior experience and resulting in less robust estimations.

To alleviate this, we propose a novel stochastic method based on a Poisson process to generate perturbative lag times for sub-trajectory sampling and use it to construct a Markov chain. Comprehensive evaluations on the double-well system, WW domain, BPTI, and RBD–ACE2 complex of SARS-CoV-2 reveal that our algorithm significantly increases the robustness and accuracy of the constructed MSM without disrupting its Markovian properties. Furthermore, the advantages of our algorithm are especially pronounced for slow dynamic modes in complex biological processes.

<!--<img src="https://github.com/microsoft/AI2BMD/blob/resources/images/markov.jpg?raw=true" width=25%>-->

- Read the Cover Story article [Stochastic Lag Time Parameterization for Markov State Models of Protein Dynamics](https://pubs.acs.org/doi/10.1021/acs.jpcb.2c03711).

- Find an application case in studying the Spike-ACE2 complex structure for the highly infectious mechanism of Omicron: [Structural insights into the SARS-CoV-2 Omicron RBD-ACE2 interaction](https://www.nature.com/articles/s41422-022-00644-8).  


## Citation
(#: co-first author; *: corresponding author)

Yusong Wang#, [Tong Wang#*](https://www.microsoft.com/en-us/research/people/watong/), Shaoning Li#, Xinheng He, Mingyu Li, [Zun Wang](https://www.microsoft.com/en-us/research/people/zunwang/), Nanning Zheng, [Bin Shao*](https://www.binshao.info/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/), Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing, Nature Communications, 15.1 (2024): 313. 

Yusong Wang#, Shaoning Li#, [Tong Wang*](https://www.microsoft.com/en-us/research/people/watong/), [Bin Shao](https://www.binshao.info/), Nanning Zheng, [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). Geometric Transformer with Interatomic Positional Encoding. NeurIPS 2023.

[Zun Wang#](https://www.microsoft.com/en-us/research/people/zunwang/), Hongfei Wu#, [Lixin Sun](https://www.microsoft.com/en-us/research/people/lixinsun/), Xinheng He, Zhirong Liu, [Bin Shao](https://www.binshao.info/), [Tong Wang*](https://www.microsoft.com/en-us/research/people/watong/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). Improving machine learning force fields for molecular dynamics simulations with fine-grained force metrics, The Journal of Chemical Physics, Volume 159, Issue 3, Cover Story.

[Tong Wang#*](https://www.microsoft.com/en-us/research/people/watong/), Xinheng He#, Mingyu Li#, [Bin Shao*](https://www.binshao.info/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). AIMD-Chig: Exploring the conformational space of a 166-atom protein Chignolin with ab initio molecular dynamics, Scientific Data 10, 549 (2023).

Shiqi Gong#, Xinheng He#, Qi Meng, Zhiming Ma, [Bin Shao*](https://www.binshao.info/), [Tong Wang*](https://www.microsoft.com/en-us/research/people/watong/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). Stochastic Lag Time Parameterization for Markov State Models of Protein Dynamics, The Journal of Physical Chemistry B 2022 126 (46), Cover Story, 2022.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE) license.

## Disclaimer

AI<sup>2</sup>BMD is a research project. It is not an officially supported Microsoft product.

## Contacts

Please contact <A href="mailto:ai2bmd@microsoft.com">AI2BMD Team</A> for any questions or suggestions. The main team members include:

- [Tong Wang](https://www.microsoft.com/en-us/research/people/watong/) (Primary contact)
- [Yatao Li](https://www.microsoft.com/en-us/research/people/yatli/)
- [Ran Bi](https://www.microsoft.com/en-us/research/people/biran/)
- [Bin Shao](https://www.binshao.info/)
- [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/)
