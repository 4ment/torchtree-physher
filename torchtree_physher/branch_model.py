from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.branch_model import SimpleClockModel as TSimpleClockModel
from torchtree.evolution.branch_model import StrictClockModel as TStrictClockModel
from torchtree.typing import ID

from .interface import Interface
from .physher import SimpleClockModel as PhysherSimpleClockModel
from .physher import StrictClockModel as PhysherStrictClockModel


class StrictClockModel(TStrictClockModel, Interface):
    def __init__(self, id_: ID, rates: AbstractParameter, tree) -> None:
        super().__init__(id_, rates, tree)
        self.inst = PhysherStrictClockModel(rates.tensor.item(), tree.inst)

    def update(self, index):
        if self._rates is not None:
            self.inst.set_parameters(self._rates.tensor[index].detach().numpy())


class SimpleClockModel(TSimpleClockModel, Interface):
    def __init__(self, id_: ID, rates: AbstractParameter, tree) -> None:
        super().__init__(id_, rates, tree)
        self.inst = PhysherSimpleClockModel(rates.tensor.tolist(), tree.inst)

    def update(self, index):
        if self._rates is not None:
            self.inst.set_parameters(self._rates.tensor[index].detach().numpy())