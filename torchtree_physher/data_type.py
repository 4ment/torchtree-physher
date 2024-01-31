from __future__ import annotations

from torchtree.evolution.datatype import GeneralDataType as TGeneralDataType
from torchtree.evolution.datatype import NucleotideDataType as TNucleotideDataType
from torchtree.typing import ID

from torchtree_physher.physher import GeneralDataType as PhysherGeneralDataType
from torchtree_physher.physher import NucleotideDataType as PhysherNucleotideDataType


class NucleotideDataType(TNucleotideDataType):
    def __init__(self, id_: ID) -> None:
        super().__init__(id_)
        self.inst = PhysherNucleotideDataType()


class GeneralDataType(TGeneralDataType):
    def __init__(self, id_: ID, codes: tuple[str, ...], ambiguities: dict) -> None:
        super().__init__(id_, codes, ambiguities)
        self.inst = PhysherGeneralDataType(codes, ambiguities)
