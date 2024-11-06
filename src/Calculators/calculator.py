from typing import Union

from ase import Atoms
from ase.calculators.calculator import Calculator, compare_atoms

from AIMD.protein import Protein


def check_state(self: Calculator, atoms: Union[Atoms, Protein], tol=1e-15):
    r"""
    Check for any system changes since last calculation.
    - Skips the check and assume no changes (and therefore no calculations), if
      'atoms.check_state' is false.
    """
    
    if getattr(atoms, "skip_check_state", False):
        return []
    else:
        return compare_atoms(self.atoms, atoms, tol=tol, excluded_properties=set(self.ignored_changes))


def patch_check_state():
    Calculator.check_state = check_state
