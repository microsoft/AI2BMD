import argparse
import os

import torch

from Calculators.device_strategy import DeviceStrategy
from utils.utils import src_dir


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

    _src_dir = src_dir()
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
        default=os.path.join(_src_dir, "ViSNet/checkpoints"),
        help="A directory including well-trained pytorch models",
    )
    parser.add_argument(
        "--ckpt-type",
        type=str,
        default="2ef43f29ec78fa5fef0b3de832bfada9",
        choices=["2ef43f29ec78fa5fef0b3de832bfada9"],
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
        "--timestep",
        type=float,
        default=1,
        help="TimeStep (fs) for simulation",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=1000,
        help="Simulation steps for simulation",
    )
    parser.add_argument(
        "--preeq-steps",
        type=int,
        default=2000,
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
        "--solvent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use solvent or not",
    )
    parser.add_argument(
        "--preprocess-method",
        type=str,
        default="FF19SB",
        choices=["FF19SB", "AMOEBA"],
        help="Method to use for preprocessing the protein",
    )
    parser.add_argument(
        "--mm-method",
        type=str,
        default="tinker-GPU",
        choices=["tinker", "tinker-GPU"],
        help="MM calculator for the nonbonded energy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fragment",
        choices=["fragment", "visnet"],
        help="""Mode for performing calculations.
        fragment=Perform fragmentation (>1 amino acids in chain).
        visnet=Feed the input directly to ViSNet.
        """,
    )
    parser.add_argument(
        "--fragment-longrange-calc",
        type=str,
        default="mm",
        choices=["mm", "pme"],
        help="Long-range interactions calculator for fragments; required for 'fragment' mode.",
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
        help="Restart the simulation",
    )
    parser.add_argument(
        "--build-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build xyz frames from the trajectory after simulation",
    )
    parser.add_argument(
        "--record-per-steps",
        type=int,
        default=100,
        help="Interval for writing out frame data",
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
        combined=Distribute work evenly amongst all fragments.
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
        "-v",
        "--verbose",
        action='count',
        default=0,
        help="""Verbosity level"""
    )

    _args = parser.parse_args(argv)
    _args.prot_name = os.path.basename(_args.prot_file)[:-4]
    if _args.log_dir is None:
        _args.log_dir = os.path.join(_args.base_dir, f"Logs-{_args.prot_name}")
    os.makedirs(_args.log_dir, exist_ok=True)
    _args.base_dir = os.path.abspath(_args.base_dir)
    _args.log_dir = os.path.abspath(_args.log_dir)
    _args.ckpt_path = os.path.abspath(_args.ckpt_path)
    _args.prot_file = os.path.abspath(_args.prot_file)
    _args.utils_dir = os.path.join(_src_dir, "utils")

    strategy_feedback = DeviceStrategy.initialize(
        _args.device_strategy,
        _args.work_strategy,
        _args.mm_method,
        torch.cuda.device_count(),
        _args.chunk_size,
    )
    _args.mm_method = strategy_feedback['preprocess-method']

    return _args

