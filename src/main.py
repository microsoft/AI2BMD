import logging
import os
import time
import warnings

import torch.multiprocessing as mp

from AIMD import arguments, envflags
from AIMD.preprocess import Preprocess
from AIMD.protein import Protein
from AIMD.simulator import SolventSimulator, NoSolventSimulator
from Calculators.calculator import patch_check_state
from utils.signals import register_print_stack_on_sigusr2
from utils.system import redir_output
from utils.utils import read_protein

if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn")

    args = arguments.init()
    if not envflags.DEBUG_RC:
        warnings.filterwarnings("ignore")
        # XXX this disables everything, not just OpenMM
        logging.disable(logging.WARNING)  # disable warnings of OpenMM
    else:
        logging.basicConfig(level=logging.DEBUG)

    logfile = os.path.join(args.log_dir, f"main-{time.strftime('%Y%m%d-%H%M%S')}.log")
    redir_output(logfile)
    register_print_stack_on_sigusr2(pass_through = True)

    preeq = Preprocess(
        prot_path=args.prot_file,
        utils_dir=args.utils_dir,
        command_save_path=os.path.join(args.log_dir, "PreprocessBackup"),
        solvent_method=args.solvent_method,
        log_dir=args.log_dir,
        temp_k=args.temp_k,
    )
    if args.solvent_method == "AMOEBA":
        (
            preeq_pdb,
            preeq_nowat_pdb,
        ) = preeq.run_preprocess()
    os.chdir(args.base_dir)

    patch_check_state()

    prot = Protein(
        read_protein(preeq_pdb),
        save_path=os.path.join(args.log_dir, "Fragment"),
        pdb4params=preeq_nowat_pdb,
    )
    
    simulator_class = SolventSimulator if args.solvent else NoSolventSimulator

    simulator = simulator_class(
        prot=prot,
        log_path=args.log_dir,
        preeq_steps=args.preeq_steps,
        temp_k=args.temp_k,
        utils_dir=args.utils_dir,
        pdb_file=preeq_pdb,
        nowat_pdb_file=preeq_nowat_pdb,
        mmcalc_type=args.mm_method,
        solvent_method=args.solvent_method,
        dev_strategy=args.device_strategy,
    )
    
    simulator.set_calculator(
        ckpt_path=args.ckpt_path,
        ckpt_type=args.ckpt_type,
        nbcalc_type=args.frag_nonbonded_calc,
    )

    simulator.simulate(
        prot_name=args.prot_name,
        simulation_steps=args.sim_steps,
        time_step=args.timestep,
        record_per_steps=1,
        hydrogen_constraints=args.constraints,
        seed=args.seed,
        restart=args.restart,
        build_frames=args.build_frames,
    )
