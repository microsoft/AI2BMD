import itertools
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from os import path as osp

import numpy as np
import torch

from AIMD import arguments
from AIMD.fragment import FragmentData
from AIMD.protein import Protein
from Calculators.combiner import DipeptideBondedCombiner
from Calculators.device_strategy import DeviceStrategy
from Calculators.visnet_calculator import ViSNetModelLike, get_visnet_model
from Fragmentation import DistanceFragment
from utils.utils import numpy_to_torch


class DLBondedCalculator:
    r"""
    DLBondedCalculator is a dipeptide bonded calculator based on
     DL calculations supported by ViSNet.
    """

    def __init__(
        self,
        ckpt_path: str,
        ckpt_type: str,
        **kwargs,
    ) -> None:
        self.models: list[ViSNetModelLike] = []
        self.ckpt_path = ckpt_path
        self.ckpt_type = ckpt_type

        # * set fragment method and combiner
        self.fragment_method = DistanceFragment()
        self.combiner = DipeptideBondedCombiner()

        print("Loading models...")
        model_path = osp.join(self.ckpt_path, f"visnet-uni-{self.ckpt_type}.ckpt")
        self.models = [
            get_visnet_model(model_path, device)
            for device in DeviceStrategy.get_bonded_devices()
        ]

    def _inference_impl(
        self, data: list[FragmentData], model: ViSNetModelLike
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return zip(*[model.dl_potential_loader(unit) for unit in data])

    def calculate(
        self, fragments: FragmentData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Calculate the energy and forces of the dipeptide.
         The basic function of DLBondedCalculator.

        Parameters:
        -----------
            fragments: FragmentData
                combined dipeptide and ACE-NME fragments.
        """

        # retrieve devices and work assignment
        devices = DeviceStrategy.get_bonded_devices()
        work = DeviceStrategy.get_work_partitions()

        # work execution
        n_devices = len(devices)
        partitions = [[] for _ in range(n_devices)]
        for idx, start, end in work:
            partitions[idx].append(fragments[start:end])

        futures: list[Future] = []
        with ThreadPoolExecutor(n_devices) as executor:
            for data, model in zip(partitions, self.models):
                futures.append(executor.submit(self._inference_impl, data, model))

        # collect results
        energy, forces = [
            np.concatenate(list(itertools.chain(*item)))
            for item in zip(*[f.result() for f in futures])
        ]

        # convert numpy arrays to torch tensors
        device = DeviceStrategy.get_default_device()

        energy = numpy_to_torch(energy, device=device)
        forces = numpy_to_torch(forces, device=device)

        # split results between dipeptides/ACE-NMEs
        dipeptides_energy, ACE_NMEs_energy = (energy[s] for s in fragments.scalar_split())
        dipeptides_forces, ACE_NMEs_forces = (forces[s] for s in fragments.vector_split())

        return (
            dipeptides_energy,
            dipeptides_forces,
            ACE_NMEs_energy,
            ACE_NMEs_forces,
        )

    def __call__(self, prot: Protein) -> tuple[np.ndarray, np.ndarray]:
        fragments = self.fragment_method.get_fragments(prot)
        (
            dipeptides_energies,
            dipeptides_forces,
            ACE_NMEs_energies,
            ACE_NMEs_forces,
        ) = self.calculate(fragments)

        energy = self.combiner.energy_combine(
            dipeptides_energies,
            ACE_NMEs_energies,
        )
        forces = self.combiner.forces_combine(
            len(prot),
            dipeptides_forces,
            ACE_NMEs_forces,
            prot.select_index,
            prot.origin_index,
        )

        return energy, forces
