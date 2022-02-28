from physherpy.interface import Interface
from physherpy.physher import HKY as PhysherHKY
from physherpy.physher import JC69 as PhysherJC69
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.substitution_model.nucleotide import HKY as THKY
from torchtree.evolution.substitution_model.nucleotide import JC69 as TJC69
from torchtree.typing import ID


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
