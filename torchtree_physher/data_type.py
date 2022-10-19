from __future__ import annotations

from torchtree.evolution.datatype import GeneralDataType as TGeneralDataType
from torchtree.typing import ID

from .physher import GeneralDataType as PhysherGeneralDataType


class GeneralDataType(TGeneralDataType):
    def __init__(self, id_: ID, codes: tuple[str, ...], ambiguities: dict) -> None:
        super().__init__(id_, codes, ambiguities)
        self.inst = PhysherGeneralDataType(codes)
