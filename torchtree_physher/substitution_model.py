from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.datatype import DataType
from torchtree.evolution.substitution_model.general import (
    GeneralNonSymmetricSubstitutionModel as TGeneralNonSymmetricSubstitutionModel,
)
from torchtree.evolution.substitution_model.nucleotide import GTR as TGTR
from torchtree.evolution.substitution_model.nucleotide import HKY as THKY
from torchtree.evolution.substitution_model.nucleotide import JC69 as TJC69
from torchtree.typing import ID

from .interface import Interface
from .physher import GTR as PhysherGTR
from .physher import HKY as PhysherHKY
from .physher import JC69 as PhysherJC69
from .physher import GeneralSubstitutionModel as PhysherGeneralSubstitutionModel
from .utils import flatten_2D


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
        kappa_flatten = flatten_2D(self._kappa.tensor)
        frequencies_flatten = flatten_2D(self._frequencies.tensor)
        self.inst = PhysherHKY(kappa_flatten[0].item(), frequencies_flatten[0].tolist())

    def update(self, index):
        kappa_flatten = flatten_2D(self._kappa.tensor)
        frequencies_flatten = flatten_2D(self._frequencies.tensor)
        self.inst.set_kappa(kappa_flatten[index])
        self.inst.set_frequencies(frequencies_flatten[index])


class GTR(TGTR, Interface):
    def __init__(
        self, id_: ID, rates: AbstractParameter, frequencies: AbstractParameter
    ) -> None:
        super().__init__(id_, rates, frequencies)
        rates_flatten = flatten_2D(self._rates.tensor)
        frequencies_flatten = flatten_2D(self._frequencies.tensor)
        self.inst = PhysherGTR(
            rates_flatten[0].tolist(), frequencies_flatten[0].tolist()
        )

    def update(self, index):
        rates_flatten = flatten_2D(self._rates.tensor)
        frequencies_flatten = flatten_2D(self._frequencies.tensor)
        self.inst.set_rates(rates_flatten[index])
        self.inst.set_frequencies(frequencies_flatten[index])


class GeneralNonSymmetricSubstitutionModel(
    TGeneralNonSymmetricSubstitutionModel, Interface
):
    def __init__(
        self,
        id_: ID,
        data_type: DataType,
        mapping: AbstractParameter,
        rates: AbstractParameter,
        frequencies: AbstractParameter,
        normalize: bool,
    ) -> None:
        super().__init__(id_, data_type, mapping, rates, frequencies, normalize)
        rates_flatten = flatten_2D(self._rates.tensor)
        frequencies_flatten = flatten_2D(self._frequencies.tensor)
        self.inst = PhysherGeneralSubstitutionModel(
            data_type.inst,
            rates_flatten[0].tolist(),
            frequencies_flatten[0].tolist(),
            mapping.tensor.tolist(),
            normalize,
        )

    def update(self, index):
        rates_flatten = flatten_2D(self._rates.tensor)
        frequencies_flatten = flatten_2D(self._frequencies.tensor)
        self.inst.set_rates(rates_flatten[index])
        if index < frequencies_flatten.shape[0]:
            self.inst.set_frequencies(frequencies_flatten[index])
