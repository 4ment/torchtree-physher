from physherpy.interface import Interface
from physherpy.physher import StrictClockModel as PhysherStrictClockModel
from physherpy.physher import TreeModel
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.branch_model import StrictClockModel as TStrictClockModel
from torchtree.typing import ID


class StrictClockModel(TStrictClockModel, Interface):
    def __init__(self, id_: ID, rates: AbstractParameter, tree: TreeModel) -> None:
        super().__init__(id_, rates, tree)
        self.inst = PhysherStrictClockModel(rates.tensor.tolist()[0])

    def update(self, index):
        if self._rates is not None:
            self._rates.set_rate(self.rates_wrapper.tensor[index])
