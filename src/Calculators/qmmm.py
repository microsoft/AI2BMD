from concurrent.futures import ThreadPoolExecutor

from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import SimpleQMMM

from AIMD.protein import Protein
from Calculators.device_strategy import DeviceStrategy
from utils.utils import WorkQueue, execution_wrapper


class AsyncQMMM(SimpleQMMM):

    qmcalc: Calculator
    mmcalc1: Calculator
    mmcalc2: Calculator
    qmatoms: Protein

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.work_queue = WorkQueue()
        self.concurrent = len(DeviceStrategy.get_solvent_devices()) > 1

    def get_qmcalc_results(self, properties, system_changes):
        # force the evaluation of results
        self.qmcalc.calculate(self.qmatoms, properties, system_changes)
        return (
            self.qmcalc.results["energy"],
            self.qmcalc.results["forces"],
        )

    def get_mmcalc1_results(self, properties, system_changes):
        # force the evaluation of results
        self.mmcalc1.calculate(self.qmatoms, properties, system_changes)
        return (
            self.mmcalc1.results["energy"],
            self.mmcalc1.results["forces"],
        )

    def get_mmcalc2_results(self, properties, system_changes):
        # force the evaluation of results
        self.mmcalc2.calculate(self.atoms, properties, system_changes)
        return (
            self.mmcalc2.results["energy"],
            self.mmcalc2.results["forces"],
        )

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.qmatoms.positions = atoms.positions[self.selection]
        if self.vacuum:
            self.qmatoms.positions += self.center - self.qmatoms.positions.mean(axis=0)

        f_args = [
            (self.get_mmcalc1_results, properties, system_changes),
            (self.get_mmcalc2_results, properties, system_changes),
        ]

        # Start some new threads to calculate the energies and forces
        with ThreadPoolExecutor(3) as executor:
            qm = executor.submit(self.get_qmcalc_results, properties, system_changes)
            mm = executor.submit(execution_wrapper, f_args, self.concurrent)

            self.work_queue.drain()

        qm_e, qm_f = qm.result()
        mm_1, mm_2 = mm.result()

        mm1_e, mm1_f = mm_1
        mm2_e, mm2_f = mm_2

        energy = mm2_e + qm_e - mm1_e
        forces = mm2_f

        if self.vacuum:
            qm_f -= qm_f.mean(axis=0)

        forces[self.selection] += qm_f - mm1_f

        self.results["energy"] = energy
        self.results["forces"] = forces
