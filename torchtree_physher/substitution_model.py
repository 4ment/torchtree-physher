from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.substitution_model.nucleotide import GTR as TGTR
from torchtree.evolution.substitution_model.nucleotide import HKY as THKY
from torchtree.evolution.substitution_model.nucleotide import JC69 as TJC69
from torchtree.typing import ID

from .interface import Interface
from .physher import GTR as PhysherGTR
from .physher import HKY as PhysherHKY
from .physher import JC69 as PhysherJC69


class JC69(TJC69, Interface):
    def __init__(self, id_: ID) -> None:
        super().__init__(id_)
        self.inst = PhysherJC69()

    def update(self, index):
        pass


class HKY(THKY, Interface):
    def __init__(
        self, id_: ID, kappa: AbstractParameter, frequencies: AbstractParameter
    ) -> None:
        super().__init__(id_, kappa, frequencies)
        self.inst = PhysherHKY(kappa.tensor.tolist()[0], frequencies.tensor.tolist())

    def update(self, index):
        self.inst.set_kappa(self._kappa.tensor[index])
        self.inst.set_frequencies(self._frequencies.tensor[index])


class GTR(TGTR, Interface):
    def __init__(
        self, id_: ID, rates: AbstractParameter, frequencies: AbstractParameter
    ) -> None:
        super().__init__(id_, rates, frequencies)
        self.inst = PhysherGTR(rates.tensor.tolist(), frequencies.tensor.tolist())

    def update(self, index):
        self.inst.set_rates(self._rates.tensor[index])
        self.inst.set_frequencies(self._frequencies.tensor[index])
