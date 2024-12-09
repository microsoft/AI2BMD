import logging
import os
import time

import torch.multiprocessing as mp

from AIMD import arguments
from AIMD.preprocess import Preprocess
from AIMD.protein import Protein
from AIMD.simulator import SolventSimulator, NoSolventSimulator
from Calculators.calculator import patch_check_state
from utils.pdb import fix_atomic_numbers, read_protein
from utils.system import redir_output

if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn")

    args = arguments.init()
    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    logfile = os.path.join(args.log_dir, f"main-{time.strftime('%Y%m%d-%H%M%S')}.log")
    redir_output(logfile)

    preeq = Preprocess(
        prot_path=args.prot_file,
        utils_dir=args.utils_dir,
        command_save_path=os.path.join(args.log_dir, "PreprocessBackup"),
        preprocess_method=args.preprocess_method,
        log_dir=args.log_dir,
        temp_k=args.temp_k,
    )
    preeq_pdb, preeq_nowat_pdb = preeq.run_preprocess()

    os.chdir(args.base_dir)

    patch_check_state()

    prot = Protein(read_protein(preeq_pdb), pdb4params=preeq_nowat_pdb)
    fix_atomic_numbers(preeq_pdb, prot)
    
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
        preprocess_method=args.preprocess_method,
        dev_strategy=args.device_strategy,
    )
    
    simulator.set_calculator(
        ckpt_path=args.ckpt_path,
        ckpt_type=args.ckpt_type,
        nbcalc_type=args.fragment_longrange_calc,
    )

    simulator.simulate(
        prot_name=args.prot_name,
        simulation_steps=args.sim_steps,
        time_step=args.timestep,
        record_per_steps=args.record_per_steps,
        hydrogen_constraints=args.constraints,
        seed=args.seed,
        restart=args.restart,
        build_frames=args.build_frames,
    )
