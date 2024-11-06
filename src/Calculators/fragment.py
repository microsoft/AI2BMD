from ase.calculators.calculator import Calculator

from Calculators.device_strategy import DeviceStrategy
from Calculators.bonded import DLBondedCalculator
from Calculators.combiner import DipeptideCombiner
from Calculators.nonbonded import MMNonBondedCalculator
from Calculators.pme import PMENonBondedCalculator
from utils.utils import WorkQueue, execution_wrapper


nonbonded_calcs = {
    'mm': MMNonBondedCalculator,
    'pme': PMENonBondedCalculator,
}

class FragmentCalculator(Calculator):
    r"""
    FragmentCalculator is a universal calculator for dipeptide fragments,
    including Bonded Calculator, QM/DL Calculator, and MM (Non-Bonded) Calculator.

    Parameters:
    -----------
        properties: list[str]
            Targets of the calculation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, is_root_calc, nbcalc_type, **kwargs):
        super().__init__(**kwargs)

        self.is_root_calc = is_root_calc

        if self.is_root_calc is True:
            self.work_queue = WorkQueue()

        # * set bonded calculator
        self.bonded_calculator = DLBondedCalculator(**kwargs)

        # * set non-bonded calculator
        device = DeviceStrategy.get_non_bonded_device()
        self.nonbonded_calculator = nonbonded_calcs[nbcalc_type](device=device)

        # * set fragment strategy
        self.concurrent = DeviceStrategy.fragment_strategy()

        # * set combiner
        self.combiner = DipeptideCombiner()

    def calculate(self, atoms, properties, system_changes):
        if self.is_root_calc:
            Calculator.calculate(self, atoms, properties, system_changes)
            self.work_queue.drain()

        f_args = [
            (self.bonded_calculator,    atoms),
            (self.nonbonded_calculator, atoms),
        ]

        energy, forces = zip(*execution_wrapper(f_args, self.concurrent))

        energy = self.combiner.energy_combine(*energy)
        forces = self.combiner.forces_combine(*forces)

        self.results = {
            "energy": energy,
            "forces": forces,
        }
