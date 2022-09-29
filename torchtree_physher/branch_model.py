from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.branch_model import SimpleClockModel as TSimpleClockModel
from torchtree.evolution.branch_model import StrictClockModel as TStrictClockModel
from torchtree.typing import ID

from .interface import Interface
from .physher import SimpleClockModel as PhysherSimpleClockModel
from .physher import StrictClockModel as PhysherStrictClockModel
from .utils import flatten_2D


class StrictClockModel(TStrictClockModel, Interface):
    def __init__(self, id_: ID, rates: AbstractParameter, tree) -> None:
        super().__init__(id_, rates, tree)
        tensor_flatten = flatten_2D(self._rates.tensor)
        self.inst = PhysherStrictClockModel(tensor_flatten[0].item(), tree.inst)

    def update(self, index):
        if self._rates is not None:
            tensor_flatten = flatten_2D(self._rates.tensor)
            self.inst.set_parameters(tensor_flatten[index].detach().numpy())


class SimpleClockModel(TSimpleClockModel, Interface):
    def __init__(self, id_: ID, rates: AbstractParameter, tree) -> None:
        super().__init__(id_, rates, tree)
        tensor_flatten = flatten_2D(self._rates.tensor)
        self.inst = PhysherSimpleClockModel(tensor_flatten[0].tolist(), tree.inst)

    def update(self, index):
        if self._rates is not None:
            tensor_flatten = flatten_2D(self._rates.tensor)
            self.inst.set_parameters(tensor_flatten[index].detach().numpy())
