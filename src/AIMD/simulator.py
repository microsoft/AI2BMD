import os
import shutil
from abc import ABC, abstractmethod

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator
from ase.constraints import Hookean
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from AIMD import arguments, envflags
from AIMD.protein import Protein
from Calculators.device_strategy import DeviceStrategy
from Calculators.fragment import FragmentCalculator
from Calculators.qmmm import AsyncQMMM
from Calculators.tinker_async import TinkerAsyncCalculator, TinkerRuntimeError
from Calculators.visnet_calculator import ViSNetCalculator
from utils.pdb import read_protein
from utils.system import get_physical_core_count
from utils.utils import (
    MDObserver,
    PDBAnalyzer,
    RNGPool,
    TemperatureRunawayError,
    WorkQueue,
)


class BaseSimulator(ABC):
    def __init__(
        self, prot: Protein, log_path: str, preeq_steps: int = 200, temp_k: int = 300
    ) -> None:
        self.prot = prot
        self.log_path = log_path
        self.simulation_save_path = os.path.join(log_path, "SimulationResults")
        os.makedirs(self.simulation_save_path, exist_ok=True)
        self.nowat_pdb = self.prot.nowater_PDB_path
        self.preeq_steps = preeq_steps
        self.prot.set_pbc(True)
        self.temp_k = temp_k

    def get_qm_idx(self):
        return list(range(len(read_protein(self.nowat_pdb))))

    def need_fragmentation(self):
        return isinstance(self.qmcalc, FragmentCalculator)

    def initialize_fragcalc(self):
        if self.need_fragmentation():
            self.qmcalc.bonded_calculator.fragment_method.fragment(self.qmatoms)
            self.qmcalc.nonbonded_calculator.set_parameters(self.qmatoms)

            start, end = self.qmatoms.fragments_start, self.qmatoms.fragments_end
        else:
            start, end = [0], [len(self.qmatoms)]

        # set work partitions based on dipeptides/ACE-NMEs
        DeviceStrategy.set_work_partitions(start, end)

    def set_calculator(self, **kwargs) -> None:
        os.chdir(self.simulation_save_path)
        self.make_calculator(**kwargs)
        self.initialize_fragcalc()

    @abstractmethod
    def make_calculator(self, **kwargs) -> Calculator:
        pass

    def make_fragment_calculator(self, is_root_calc: bool, **kwargs) -> FragmentCalculator:
        mode = arguments.get().mode
        if mode == "fragment":
            return FragmentCalculator(is_root_calc=is_root_calc, **kwargs)
        if mode == "visnet":
            return ViSNetCalculator(is_root_calc=is_root_calc, **kwargs)

    def simulate(
        self, prot_name: str, simulation_steps: int, time_step: float, 
        record_per_steps: int, hydrogen_constraints: bool,
        seed: int, restart: bool, build_frames: bool
    ):
        restart_traj_path = os.path.join(self.log_path, f"{prot_name}-traj.traj")

        if restart:
            with Trajectory(restart_traj_path) as ori_traj:
                restart_frame_count = len(ori_traj)
                restart_last_frame = ori_traj[-1]
                self.prot.set_positions(restart_last_frame.get_positions())
                self.prot.set_velocities(restart_last_frame.get_velocities())
        else:
            restart_frame_count = 0
            MaxwellBoltzmannDistribution(self.prot, temperature_K=self.temp_k, rng=np.random.RandomState(seed))

        '''
        MolDyn = NVTBerendsen(
            self.prot,
            timestep=time_step * units.fs,
            temperature=self.temp_k,
            taut=0.01 * 1000 * units.fs,
        )
        '''

        # initialize rng pool
        rng_pool = RNGPool(seed=seed, shape=(len(self.prot), 3), count=2)

        MolDyn = Langevin(
            self.prot,
            timestep=time_step * units.fs,
            temperature_K=self.temp_k,
            friction=0.001 / units.fs,
            rng=rng_pool,
        )

        if restart: 
            moldyn_traj_filename = os.path.join(self.log_path, f"{prot_name}-traj-restart.traj")
        else:
            moldyn_traj_filename = os.path.join(self.log_path, f"{prot_name}-traj.traj")

        moldyn_traj = Trajectory(moldyn_traj_filename, "w", self.prot)

        observer = MDObserver(
            a=self.prot,
            q=self.qmatoms,
            md=MolDyn,
            traj=moldyn_traj,
            rng=rng_pool,
            step_offset=restart_frame_count,
            temp_k=self.temp_k,
        )
        MolDyn.attach(observer.save_traj_copy, interval=record_per_steps)
        MolDyn.attach(observer.write_traj, interval=record_per_steps)
        MolDyn.attach(observer.printenergy, interval=record_per_steps)
        MolDyn.attach(observer.fill_rng_pool, interval=1)

        if (not restart) and (self.preeq_steps != 0):
            init_constraint = self.prot.constraints.copy()
            indices_to_constrain = self.get_qm_idx()
            restraints = [10, 5, 1, 0.5, 0.1]
            print("Start pre-equilibration")
            for restraint in restraints:
                print(
                    f"Pre-equilibration with {restraint} eV/A² for {self.preeq_steps} steps"
                )
                constraints = []
                ref_positions = self.prot.positions
                for idx in indices_to_constrain:
                    pos = ref_positions[idx]
                    kcalmol2ev = (units.kcal / units.mol) / units.eV
                    constraint = Hookean(a1=idx, a2=pos, k=restraint * kcalmol2ev, rt=0)
                    constraints.append(constraint)
                self.prot.constraints.extend(constraints)

                try:
                    MolDyn.run(self.preeq_steps)
                except TemperatureRunawayError:
                    print("Thermostat detects a temperature runaway condition, cannot proceed.")
                    exit(-1)
                except TinkerRuntimeError:
                    print("Solvent dynamic component Tinker terminated abnormally, cannot proceed.")
                    exit(-1)
                self.prot.constraints = init_constraint.copy()
            print("Pre-equilibration finished!")

        if hydrogen_constraints is True:
            pdb_analyzer = PDBAnalyzer(self.nowat_pdb)
            hydrogen_bonds = pdb_analyzer.find_bonded_atoms("H")
            hydrogen_constraints = []

            for pair in hydrogen_bonds:
                # * Hookean constraints
                hydrogen_constraint = Hookean(
                    a1=pair[0], a2=pair[1], k=pair[3], rt=pair[2]
                )
                hydrogen_constraints.append(hydrogen_constraint)

            self.prot.constraints.extend(hydrogen_constraints)

        if restart:
            print(f"Re-start simulation for {simulation_steps} steps")
        else:
            print(f"Start simulation for {simulation_steps} steps")
            
        try:
            MolDyn.run(simulation_steps)
        except TinkerRuntimeError:
            print("Solvent dynamic component Tinker terminated abnormally, cannot proceed.")
            exit(-1)
        except TemperatureRunawayError:
            print("Thermostat detects a temperature runaway condition, cannot proceed.")
            exit(-1)

        print("Simulation finished!")
        WorkQueue.finalise()
        moldyn_traj.close()

        if build_frames and not restart:
            self.build_frames_from_traj(prot_name, record_per_steps, MolDyn.nsteps)

        if not envflags.DEBUG_RC:
            shutil.rmtree(os.path.join(self.log_path, "SimulationResults"))

    def build_frames_from_traj(self, prot_name, record_per_steps, nsteps):
        print("Building frames from trajectory...")
        simutraj = Trajectory(os.path.join(self.log_path, f"{prot_name}-traj.traj"))
        opt_traj_filename = os.path.join(self.log_path, f"{prot_name}-traj.xyz")
        os.makedirs(os.path.join(self.log_path, "frames"), exist_ok=True)
        # break the trajectory into frames (xyz), then append them all to opt_traj_filename
        for i in range(0, nsteps, record_per_steps):
            atoms = simutraj[i]
            frame_filename = os.path.join(self.log_path, "frames", f"structure{i:0>5}.xyz")
            write(frame_filename, atoms)
            with open(frame_filename) as finframe:
                inframe = finframe.read()
            with open(opt_traj_filename, "a") as fopt_traj:
                fopt_traj.write(inframe)

        os.chdir(self.log_path)
        os.makedirs("results", exist_ok=True)
        os.system(f"cp {prot_name}-traj.xyz results")
        print("Done building frames from trajectory.")


class SolventSimulator(BaseSimulator):
    def __init__(
        self,
        prot: Protein,
        log_path: str,
        preeq_steps: int,
        temp_k: int,
        utils_dir: str,
        pdb_file: str,
        nowat_pdb_file: str,
        mmcalc_type: str,
        preprocess_method: str,
        dev_strategy: str,
    ) -> None:
        super().__init__(prot, log_path, preeq_steps, temp_k)

        self.utils_dir = utils_dir
        self.pdb_file = pdb_file
        self.nowat_pdb_file = nowat_pdb_file
        self.mmcalc_type = mmcalc_type
        self.preprocess_method = preprocess_method
        self.dev_strategy = dev_strategy

    def make_mm_calculator(self):
        devices = DeviceStrategy.get_solvent_devices()
        
        if self.mmcalc_type in ['tinker', 'tinker-GPU']:
            mm_calc = TinkerAsyncCalculator(
                pdb_file=self.pdb_file,
                utils_dir=self.utils_dir,
                devices=devices,
            )
        else:
            raise ValueError(f"Unknown mm calculator: {self.mmcalc_type}")
        return mm_calc

    def make_mm_qmregion_calculator(self):
        devices = DeviceStrategy.get_solvent_devices()
        if self.mmcalc_type in ['tinker', 'tinker-GPU']:
            mm_qmregion_calc = TinkerAsyncCalculator(
                pdb_file=self.nowat_pdb_file,
                utils_dir=self.utils_dir,
                devices=devices,
            )
        else:
            raise ValueError(f"Unknown mm calculator: {self.mmcalc_type}")
        return mm_qmregion_calc

    def make_calculator(self, **kwargs):
        self.prot.calc = AsyncQMMM(
            selection=self.get_qm_idx(),
            qmcalc=self.make_fragment_calculator(is_root_calc=False, **kwargs),
            mmcalc1=self.make_mm_qmregion_calculator(),
            mmcalc2=self.make_mm_calculator(),
        )

        self.prot.calc.initialize_qm(self.prot)

        self.qmcalc = self.prot.calc.qmcalc
        self.qmatoms = self.prot.calc.qmatoms

        if isinstance(self.prot.calc.mmcalc1, TinkerAsyncCalculator):
            self.prot.calc.mmcalc1.atoms = self.qmatoms
            self.prot.calc.mmcalc1._start_tinker()
        if isinstance(self.prot.calc.mmcalc2, TinkerAsyncCalculator):
            self.prot.calc.mmcalc2.atoms = self.prot
            self.prot.calc.mmcalc2._start_tinker()


class NoSolventSimulator(BaseSimulator):
    def __init__(
        self,
        prot: Protein,
        log_path: str,
        preeq_steps: int,
        temp_k: int,
        **kwargs
    ) -> None:
        super().__init__(prot, log_path, preeq_steps, temp_k)

        self.prot = self.prot[self.get_qm_idx()]


    def make_calculator(self, **kwargs):
        self.prot.calc = self.make_fragment_calculator(is_root_calc=True, **kwargs)

        self.qmcalc = self.prot.calc
        self.qmatoms = self.prot
