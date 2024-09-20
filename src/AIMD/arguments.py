import argparse
import os
from os import path as osp

import torch
from Calculators.device_strategy import DeviceStrategy
import utils


_args = None


def get():
    if not _args:
        raise Exception("Arguments are not initialized. Call initialize() first.")
    return _args


def init(argv=None):
    """Initializes the argument registry. If no argv is supplied (default), 
    parses arguments from process command line.
    The initialization result will be kept in the module-level member `_args`,
    so that the settings can be retrieved from other modules with get().
    """
    global _args

    _src_dir = utils.src_dir()
    parser = argparse.ArgumentParser(description="DL Molecular Simulation.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.getcwd(),
        help="A directory for running simulation",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="A directory for saving results",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=osp.join(_src_dir, "ViSNet/checkpoints"),
        help="A directory including well-trained pytorch models",
    )
    parser.add_argument(
        "--ckpt-type",
        type=str,
        default="e99be6c80110870c69c60958586cec44",
        choices=["e99be6c80110870c69c60958586cec44"],
        help="Checkpoint type, which is the md5sum of the model checkpoint file",
    )
    parser.add_argument(
        "--prot-file",
        type=str,
        required=True,
        help="Protein file for simulation",
    )
    parser.add_argument(
        "--temp-k",
        type=int,
        default=300,
        help="Simulation temperature in Kelvin",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=100,
        help="Simulation steps for simulation",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1,
        help="TimeStep (fs) for simulation",
    )
    parser.add_argument(
        "--preeq-steps",
        type=int,
        default=100,
        help="Pre-equilibration simulation steps for each constraint",
    )
    parser.add_argument(
        "--max-cyc",
        type=int,
        default=100,
        help="Maximum energy minimization cycles in preprocessing",
    )
    parser.add_argument(
        "--constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain hydrogen bonds",
    )
    parser.add_argument(
        "--mm-method",
        type=str,
        default="tinker-GPU",
        choices=["tinker", "tinker-GPU"],
        help="MM calculator for the nonbonded energy",
    )
    parser.add_argument(
        "--solvent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use solvent or not",
    )
    parser.add_argument(
        "--solvent-method",
        type=str,
        default="AMOEBA",
        choices=["AMOEBA"],
        help="Method to use for preprocessing the protein",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for simulation",
    )
    parser.add_argument(
        "--restart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="restart the simulation or not",
    )
    parser.add_argument(
        "--build-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="build xyz frames from the trajectory after simulation",
    )
    parser.add_argument(
        "--device-strategy",
        type=str,
        default="small-molecule",
        choices=["excess-compute", "small-molecule", "large-molecule"],
        help="""The compute device allocation strategy.
        excess-compute=Assume compute resources are more than sufficient for
                ViSNet inference. Reserves last GPU for solvent/non-bonded
                computation.
        small-molecule=Maximise resources for ViSNet.
        large-molecule=Maximise resources for ViSNet, while also maximising
                concurrency and usage of GPUs for computation.
        """,
    )
    parser.add_argument(
        "--work-strategy",
        type=str,
        default="combined",
        choices=["combined"],
        help="""The work allocation strategy.
        combined=Distribute work evenly amongst both types of fragments.
        """,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=9999,
        help="""Define the maximum chunk size (in units of atoms) for
        ACE-NME/dipeptide fragments.  The data will be split and processed
        according to these sizes.
        """,
    )
    parser.add_argument(
        "--frag-nonbonded-calc",
        type=str,
        default="mm",
        choices=["mm", "pme"],
        help="Nonbonded calculator for fragments; required when fragmentation is enabled.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='count',
        default=0,
        help="""Verbosity level"""
    )

    _args = parser.parse_args(argv)
    _args.prot_name = osp.basename(_args.prot_file)[:-4]
    if _args.log_dir is None:
        _args.log_dir = osp.join(_args.base_dir, f"Logs-{_args.prot_name}")
    os.makedirs(_args.log_dir, exist_ok=True)
    _args.base_dir = osp.abspath(_args.base_dir)
    _args.log_dir = osp.abspath(_args.log_dir)
    _args.ckpt_path = osp.abspath(_args.ckpt_path)
    _args.prot_file = osp.abspath(_args.prot_file)
    _args.utils_dir = osp.join(_src_dir, "utils")

    strategy_feedback = DeviceStrategy.initialize(
        _args.device_strategy,
        _args.work_strategy,
        _args.mm_method,
        torch.cuda.device_count(),
        _args.chunk_size,
    )
    _args.mm_method = strategy_feedback['solvent-method']

    return _args

