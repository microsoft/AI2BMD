import bisect
import logging
import math
import subprocess

import torch

from AIMD.fragment import FragmentInfo
from utils.system import get_physical_core_count


class DeviceStrategy:
    """Computation resource is required by:
    - Preprocess
    - Simulator (Solvent/Non-solvent)
        - FragmentCalculator
            - Bonded calculator
            - Non-bonded calculator
        - Solvent-related calculators
            - MM calculator
            - MM/QM region calculator
    """

    @classmethod
    def _check_device(cls, device: str):
        if device == 'cpu':
            return
        elif device.startswith('cuda'):
            tup = device.split(':')
            assert len(tup) == 2, "invalid device syntax"
            n = int(tup[1])
            assert n >= 0 and n < cls._gpu_count, "invalid device index"
            return
        else:
            raise Exception("Unrecognized device")


    @classmethod
    def get_preprocess_device(cls):
        return cls._preprocess_device
    
    
    @classmethod
    def get_bonded_devices(cls):
        if len(cls._bonded_devices) < 1:
            raise Exception("No compute resources for bonded calculation")
        for dev in cls._bonded_devices:
            cls._check_device(dev)
        return cls._bonded_devices


    @classmethod
    def get_non_bonded_device(cls):
        cls._check_device(cls._non_bonded_device)
        return cls._non_bonded_device


    @classmethod
    def get_solvent_devices(cls):
        if len(cls._solvent_devices) < 1:
            raise Exception("No compute resources for solvent calculation")
        for dev in cls._solvent_devices:
            cls._check_device(dev)
        return cls._solvent_devices


    @classmethod
    def get_optimiser_device(cls):
        return cls._optimiser_device


    @classmethod
    def get_default_device(cls):
        cls._check_device(cls._default_device)
        return cls._default_device


    @classmethod
    def fragment_strategy(cls):
        return cls._fragment_strategy


    @classmethod
    def _set_combined_work_partitions(cls, devices: list[int], start: list[int], end: list[int]):
        """
        Work partition strategy, where the combined work for
        ACE-NMEs/dipeptides is split evenly.
        """
        partitions = []

        n_blocks = len(devices)
        a_end = len(start)

        chunk = cls._chunk_size

        # divide work into blocks, in units of atoms
        b_prev = 0
        for i in range(n_blocks):
            block = (end[-1] - start[b_prev]) // (n_blocks - i)
            b_end = bisect.bisect(start, block + start[b_prev])
            b_idx = b_end - 1

            block_end = block + start[b_prev]
            if (block_end - start[b_idx]) < (end[b_idx] - block_end):
                b_end = b_end - 1

            b_end = min(b_end, a_end)

            # divide work into chunks
            c_prev = b_prev
            while c_prev != b_end:
                c_end = bisect.bisect(start, chunk + start[c_prev])
                c_idx = c_end - 1

                chunk_end = chunk + start[c_prev]
                if (chunk_end - start[c_idx]) < (end[c_idx] - chunk_end):
                    c_end = c_end - 1

                c_end = min(c_end, b_end)

                partitions.append((i, c_prev, c_end))

                c_prev = c_end

            b_prev = b_end

        cls._work_partitions = partitions


    @classmethod
    def set_work_partitions(cls, start: list[int], end: list[int]):
        bonded = cls._bonded_devices

        cls._set_combined_work_partitions(bonded, start, end)


    @classmethod
    def get_work_partitions(cls):
        return cls._work_partitions


    @classmethod
    def initialize(cls, dev_strategy: str, work_strategy: str, preprocess_method: str, gpu_count: int, chunk_size: int):
        cls._gpu_count = gpu_count
        cls._dev_strategy = dev_strategy
        cls._work_strategy = work_strategy
        cls._chunk_size = chunk_size

        cls._fragment_strategy = False

        except_last = range(gpu_count - 1)
        all_gpus = range(gpu_count)
        last_gpu = gpu_count - 1

        # device slots
        preprocess = "cpu"
        bonded = []
        non_bonded = "cpu" if gpu_count == 0 else f"cuda:{last_gpu}"
        solvent = []
        default = "cpu" if gpu_count == 0 else "cuda:0"
        optimiser = "cpu"
        
        if preprocess_method == "tinker-GPU" and gpu_count > 0:
            preprocess = f"cuda:{last_gpu}"
        else:
            preprocess = "cpu"

        print(f"DeviceStrategy: setting strategy to [{dev_strategy} / {work_strategy}]")

        if work_strategy == "combined" and chunk_size == 0:
            raise ValueError(f"chunk-size: {chunk_size} must be non-zero for 'combined' work strategy")

        # see: AIMD/arguments.py: --device-strategy for docs
        if dev_strategy == 'excess-compute':
            if gpu_count == 0:
                bonded = ["cpu", "cpu"]
            elif gpu_count == 1:
                bonded = ["cuda:0"]
            else:
                bonded = [f"cuda:{i}" for i in except_last]

            if gpu_count > 0:
                solvent = [f"cuda:{last_gpu}"]

        elif dev_strategy == 'small-molecule':
            if gpu_count == 0:
                bonded = ["cpu", "cpu"]
            else:
                bonded = [f"cuda:{i}" for i in all_gpus]

            if gpu_count > 1:
                solvent = ["cuda:1", "cuda:0"]
            elif gpu_count > 0:
                solvent = ["cuda:0"]

        elif dev_strategy == 'large-molecule':
            if gpu_count == 0:
                bonded = ["cpu", "cpu"]
            else:
                bonded = [f"cuda:{i}" for i in all_gpus]

            if gpu_count > 3:
                solvent = ["cuda:2", "cuda:1"]
            elif gpu_count > 2:
                solvent = ["cuda:1"]
            elif gpu_count > 0:
                solvent = ["cuda:0"]

            if gpu_count > 2:
                optimiser = "cuda:0"

        else:
            raise Exception("Unknown compute strategy")

        if dev_strategy == 'large-molecule':
            # run bonded/non-bonded calculations concurrently
            cls._fragment_strategy = True

        if preprocess_method == "tinker-GPU":
            if len(solvent) == 0:
                logging.error("tinker-GPU is specified, but there's no GPU. Reverting back to CPU.")
                solvent = ["cpu"]
                preprocess_method = "tinker"
        else:
            solvent = ["cpu"]

        cls._bonded_devices = bonded
        cls._non_bonded_device = non_bonded
        cls._solvent_devices = solvent
        cls._optimiser_device = optimiser
        cls._default_device = default
        cls._preprocess_device = preprocess

        cls._work_partitions = []

        # On some machines, libtorch.so->libgomp.so excessively consume CPU resource, saturating all cores
        # and bring down performance, due to heavy synchronization.
        # We need to manually tell torch to start less CPU threads in this case.
        libgomp_bug_cpu_blacklist = [
            "Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz"
        ]
        lscpu = subprocess.Popen("lscpu", stdout=subprocess.PIPE)
        if lscpu.wait():
            raise RuntimeError("lscpu")
        output = lscpu.stdout.read().decode().splitlines()
        cpu_model = next(x for x in output if "Model name:" in x).split(':')[1].strip()
        # blacklist is weak against unknown CPU models, especially in Azure VMs (they upgrade all the time).
        # so currently we don't rely on this check.
        bad_cpu = cpu_model in libgomp_bug_cpu_blacklist
        # TODO check kernel and libgomp versions

        # setting num threads to 1 will only hurt performance only if the model is running on CPU
        model_on_cpu = "cpu" in bonded
        if not model_on_cpu:
            torch.set_num_threads(1)
        else:
            # split the CPU resources to models
            total_threads = get_physical_core_count()
            total_models = sum([1 for x in bonded if x == 'cpu'])
            # this shouldn't happen but going defensive anyways
            total_models = max(1, total_models)
            torch_threads = max(1, total_threads // total_models)
            torch.set_num_threads(torch_threads)

        return { 'preprocess-method': preprocess_method }
