# AI<sup>2</sup>BMD: AI-powered *ab initio* biomolecular dynamics simulation

## Contents

- [Overview](#overview)
- [Get Started](#get-started)
- [Datasets](#datasets)
- [System Requirements](#system-requirements)
- [Advanced Setup](#advanced-setup)
- [Related Research](#related-research)
- [Citation](#citation)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Contacts](#contacts)

## Overview

AI<sup>2</sup>BMD is a program for efficiently simulating protein molecular dynamics with *ab initio* accuracy. This repository contains the simulation program, datasets, and public materials related to AI<sup>2</sup>BMD. The main content of AI<sup>2</sup>BMD is published on [Nature](https://www.nature.com/articles/s41586-024-08127-z).

Here is an animation to illustrate how AI<sup>2</sup>BMD works.

https://github.com/user-attachments/assets/912a3e5a-c465-4dc7-8c2d-9f7807cac2a7



## Get Started

The source code of AI<sup>2</sup>BMD is hosted in this repository.
We package the source code and runtime libraries into a Docker image, and provide a Python launcher program to simplify the setup process.
To run the simulation program, you don't need to clone this repository. Simply download `scripts/ai2bmd` and launch it (Python >=3.7 and docker enviroments are required).


We can run a molecular dynamics simulation as follows.

```shell
# skip the following two lines if you've already set up the launcher
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/scripts/ai2bmd'
chmod +x ai2bmd
# download the Chignolin protein structure data file
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/examples/chig.pdb'
# download the preprocessed and solvated Chignolin protein structure data files
wget --directory-prefix=chig_preprocessed 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/examples/chig_preprocessed/chig-preeq.pdb'
wget --directory-prefix=chig_preprocessed 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/examples/chig_preprocessed/chig-preeq-nowat.pdb'
# launch the program, with all simulation parameters set to default values
# you may need to "sudo" the following line if the docker group is not configured for the user
./ai2bmd --prot-file chig.pdb --preprocess-dir chig_preprocessed --preeq-steps 0 --sim-steps 1000 --record-per-steps 1
```

Here we use a very simple protein `Chignolin` as an example.
The program will run a simulation with the default parameters.

The results will be placed in a new directory `Logs-chig`.
The directory contains the simulation trajectory file:

- chig-traj.traj: The full trajectory file in ASE binary format.

Note: Currently, AI<sup>2</sup>BMD supports MD simulations for proteins with neutral terminal caps (ACE and NME), single chain and standard amino acids.



## Datasets

### Protein Unit Dataset

The Protein Unit Dataset covers about 20 million conformations for dipeptides calculated at DFT level. It can be downloaded with the following commands:

```shell
# skip the following two lines if you've already set up the launcher
wget 'https://raw.githubusercontent.com/microsoft/AI2BMD/main/scripts/ai2bmd'
chmod +x ai2bmd
# you may need to "sudo" the following line if the docker group is not configured for the user
./ai2bmd --download-training-data
```

When it finishes, the current working directory will be populated by the numpy data files (*.npz).

### AIMD-Chig Dataset

The AIMD-Chig dataset consists of 2 million conformations of the 166-atom `Chignolin`, along with their corresponding potential energy and atomic forces calculated using Density Functional Theory (DFT) at the M06-2X/6-31G* level.

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


## Advanced Setup
### Environment
The runtime libraries and requirents are packed into a Docker image for convenience and practicality. Before launching the Docker image, you need to install the Docker software (see https://docs.docker.com/engine/install/ for more details) and add the user to docker group with the following commands:

```shell
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### Protein File Preparation

The input file for AI<sup>2</sup>BMD should be `.pdb` format.
If hydrogen atoms are missing in the `.pdb` file, hydrogens should be added.
Then, the protein should be capped with ACE (acetyl) at the N-terminus and NME (N-methyl) at the C-terminus.  These steps can be efficiently done using the PyMOL software with the following commands as a reference.

```python
from pymol import cmd
pymol.finish_launching()
cmd.load("your_protein.pdb","molecule")
cmd.h_add("molecule") # Adding hydrogen

cmd.wizard("mutagenesis")
cmd.get_wizard().set_n_cap("acet")
selection = "/%s//%s/%s" % (molecule, chain, resi) #selection of N-term
cmd.get_wizard().do_select(selection)
cmd.get_wizard().apply()

cmd.get_wizard().set_c_cap("nmet")
selection = "/%s//%s/%s" % (molecule, chain, resi) #selection of N-term
cmd.get_wizard().do_select(selection)
cmd.get_wizard().apply()

cmd.set_wizard()
```

Next, you can use AmberTools' `pdb4amber` utility to adjust atom names in the `.pdb` file, specifically ensuring compatibility for ACE and NME as required by `ai2bmd`. The atom names for ACE and NME should conform to the following:

- ACE: C, O, CH3, H1, H2, H3
- NME: N, CH3, H, HH31, HH32, HH33

```
pdb4amber -i your_protein.pdb -o processed_your_protein.pdb
```

In addition, please verify that there are no `TER` separators in the protein chain. Additionally, the residue numbering should start from 1 without gaps.


After completing the above steps, your `.pdb` file should resemble the following format:

```
ATOM      1  H1  ACE     1      10.845   8.614   5.964  1.00  0.00           H
ATOM      2  CH3 ACE     1      10.143   9.373   5.620  1.00  0.00           C
ATOM      3  H2  ACE     1       9.425   9.446   6.437  1.00  0.00           H
ATOM      4  H3  ACE     1       9.643   9.085   4.695  1.00  0.00           H
ATOM      5  C   ACE     1      10.805  10.740   5.408  1.00  0.00           C
ATOM      6  O   ACE     1      10.682  11.417   4.442  1.00  0.00           O
...
ATOM    170  N   NME    12       9.499   8.258  10.367  1.00  0.00           N
ATOM    171  H   NME    12       9.393   8.028  11.345  1.00  0.00           H
ATOM    172  CH3 NME    12       8.845   7.223   9.569  1.00  0.00           C
ATOM    173 HH31 NME    12       7.842   6.990   9.925  1.00  0.00           H
ATOM    174 HH32 NME    12       8.798   7.589   8.543  1.00  0.00           H
ATOM    175 HH33 NME    12       9.418   6.305   9.435  1.00  0.00           H
END

```

You can also take the protein files in `examples` folder as reference. Note, currently, the machine learning potential doesn't support the protein with disulfide bonds well. We will update it soon.

### Preprocess
During the preprocess, the solvated sytem is built and encounted energy minimization and alternative pre-equilibrium stages. Currently, AI<sup>2</sup>MD provides two methods for the preprocess via the argument `preprocess_method`.

If you choose the `FF19SB` method, the system will go through solvation, energy minimization, heating and several pre-equilibrium stages. To accelerate the preprocess by multiple CPU cores and GPUs, you should get AMBER software packages and modify the corresponding commands in `src/AIMD/preprocess.py`.

If you choose the `AMOEBA` method, the system will go through solvation and energy minimization stages. We highly recommend to perform pre-equilibrium simulations to let the simulation system fully relaxed.

### Simulation
AI<sup>2</sup>BMD provides two modes for performing the production simulations via the argument `mode`. The default mode of `fragment` represents protein is fragmented into dipeptides and then calculated by the machine learning potential in every simulation step.

AI<sup>2</sup>BMD also supports to train the machine learning potential by yourselves and perform simulations without fragmentation. The `visnet` mode represents the potential energy and atomic forces of the protein are directly calculated by the ViSNet model as a whole molecule without fragmentation. When using this mode, you need to train ViSNet model with the data of the molecules by yourself, upload the model to `src/ViSNet` and give the corresponding value to the argument `ckpt-type`. In this way, you can use AI<sup>2</sup>BMD simulation program to simulate any kinds of molecules beyond proteins. To train the ViSNet model by yourselves, please check out the branch [ViSNet](https://github.com/microsoft/AI2BMD/tree/ViSNet) for the source code, instructions on model training, and more techniqucal details.

To perform the whole AI<sup>2</sup>BMD simulation including the preprocess, please use the following commands as reference.

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
#   --base-dir path/to/base-dir    Directory for running simulation (default: current directory)
#   --log-dir  path/to/log-dir     Directory for logs, results (default: base-dir/Logs-protein-name)
#   --src-dir  path/to/src-dir     Mount src-dir in place of src/ from this repository (default: not used)
#
# [Simulation parameter options]
#   --sim-steps nnn                Simulation steps
#   --temp-k nnn                   Simulation temperature in Kelvin
#   --timestep nnn                 Time-step (fs) for simulation
#   --preeq-steps nnn              Pre-equilibration simulation steps for each constraint
#   --max-cyc nnn                  Maximum energy minimization cycles in preprocessing
#   --preprocess-method [method]   The method for preprocess
#   --mode [mode]                  Use fragmentation or not during the simulation
#   --record-per-steps nnn         The frequency to save trajectory
#
# [Performance tweaks]
#   --device-strategy [strategy]   The compute device allocation strategy
#       excess-compute                 Reserves last GPU for non-bonded/solvent computation
#       small-molecule                 Maximize resources for model inference
#       large-molecule                 Improve performance for large molecules
#   --chunk-size nnn               Number of atoms in each batch (reduces memory consumption)
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

### Post-analysis
The format of the simulation trajectory is `.traj` of ASE format. To convert it to `.dcd` format for visualization, you can install MDAnalysis first and take `src/utils/traj2dcd.py` as reference with the following commands:

```shell
python traj2dcd.py --input xxx.pdb --output xxx.dcd --pdb xxx.pdb --num-atoms nnn --stride nnn

# arguments
# --input         The name of the input trajectory file
# --output        The name of the output trajectory file
# --pdb           The reference pdb file corresponding to the input trajectory
# --num-atoms     The number of atoms for protein or the whole solvated system
# --stride        The frequency to output the trajectory
```

### Trouble shooting
The simulations may collapse due to insufficient modeling on proton hopping or improper simulation system settings. Proton hopping occurs frequently, especially for large biomolecules and long simulations. Since there are a few cases of proton hopping during machine learning potential training process, the model may encounter "out-of-distribution" cases, giving incorrect atomic forces and resulting in a collapse of the simulation. We will continuously update the AI<sup>2</sup>BMD potential with more powerful prediction ability and also highly recommend anyone to contribute to the dataset for model finetuning.

To avoid and alleviate simulation collapse, we provide some suggestions: 1) fully relax the simulation system before the production simulation runs; 2) increase the duration of the preequilibrium simulations; 3) increase the duration of simulations with constraints in the production runs (via the argument `preeq-steps`); 4) restart the simulation from a few steps before the crash; 5) increase the box size of solvent; 6) adjust other simulation system setting.

Beyond directly performing simulations, we also encourage users to employ AI<sup>2</sup>BMD for reweighting the existing simulation trajectories and calculate protein properties accordingly.

## Related Research

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

Tong Wang#\*, Xinheng He#, Mingyu Li#, Yatao Li#, Ran Bi, Yusong Wang, Chaoran Cheng, Xiangzhen Shen, Jiawei Meng, He Zhang, Haiguang Liu, Zun Wang, Shaoning Li, Bin Shao\*, Tie-Yan Liu. Ab initio characterization of protein molecular dynamics with AI<sup>2</sup>BMD. Nature 2024.

Yusong Wang#, Tong Wang#\*, Shaoning Li#, Xinheng He, Mingyu Li, Zun Wang, Nanning Zheng, Bin Shao*, Tie-Yan Liu, Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing, Nature Communications, 15.1 (2024): 313.

Tong Wang#\*, Xinheng He#, Mingyu Li#, Bin Shao*, Tie-Yan Liu. AIMD-Chig: Exploring the conformational space of a 166-atom protein Chignolin with ab initio molecular dynamics, Scientific Data 10, 549 (2023).

Yusong Wang#, Shaoning Li#, Tong Wang*, Bin Shao, Nanning Zheng, Tie-Yan Liu. Geometric Transformer with Interatomic Positional Encoding. NeurIPS 2023.

Zun Wang#, Hongfei Wu#, Lixin Sun, Xinheng He, Zhirong Liu, Bin Shao, Tong Wang*, Tie-Yan Liu. Improving machine learning force fields for molecular dynamics simulations with fine-grained force metrics, The Journal of Chemical Physics, Volume 159, Issue 3, Cover Story.

Shiqi Gong#, Xinheng He#, Qi Meng, Zhiming Ma, Bin Shao*, Tong Wang*, Tie-Yan Liu. Stochastic Lag Time Parameterization for Markov State Models of Protein Dynamics, The Journal of Physical Chemistry B 2022 126 (46), Cover Story, 2022.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE) license.

## Disclaimer

AI<sup>2</sup>BMD is a research project. It is not an officially supported Microsoft product.

## Contacts

Please contact <A href="mailto:tongwang.bio@outlook.com">Tong Wang</A> (Project Lead) and <A href="mailto:biran@microsoft.com">Ran Bi</A> for any questions, suggestions, and technical support.
