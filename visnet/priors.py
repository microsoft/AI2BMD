from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_warn

__all__ = ["Atomref"]


class BasePrior(nn.Module, metaclass=ABCMeta):
    """
    Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.Atomref` prior.
    """

    def __init__(self):
        super(BasePrior, self).__init__()

    @abstractmethod
    def get_init_args(self):
        """
        A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
        return

    @abstractmethod
    def forward(self, x, z):
        """
        Forward method of the prior model.

        Args:
            x (torch.Tensor): scalar atomwise predictions from the model.
            z (torch.Tensor): atom types of all atoms.

        Returns:
            torch.Tensor: updated scalar atomwise predictions
        """
        return


class Atomref(BasePrior):
    """
    Atomref prior model.
    When using this in combination with some dataset, the dataset class must implement
    the function `get_atomref`, which returns the atomic reference values as a tensor.
    """

    def __init__(self, max_z=None, dataset=None):
        super(Atomref, self).__init__()
        if max_z is None and dataset is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if dataset is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = dataset.get_atomref()
            if atomref is None:
                rank_zero_warn(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. "
                    "atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)

    def get_init_args(self):
        return dict(max_z=self.initial_atomref.size(0))

    def forward(self, x, z):
        return x + self.atomref(z)
