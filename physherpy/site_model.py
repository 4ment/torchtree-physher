from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.site_model import (
    ConstantSiteModel as TorchtreeConstantSiteModel,
)
from torchtree.evolution.site_model import WeibullSiteModel as TorchtreeWeibullSiteModel
from torchtree.typing import ID

from physherpy.interface import Interface
from physherpy.physher import ConstantSiteModel as PhysherConstantSiteModel
from physherpy.physher import WeibullSiteModel as PhysherWeibullSiteModel


class ConstantSiteModel(TorchtreeConstantSiteModel, Interface):
    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_, mu)
        self.inst = PhysherConstantSiteModel(None)

    def update(self, index):
        if self._mu is not None:
            self.inst.set_mu(self._mu.tensor[index])


class WeibullSiteModel(TorchtreeWeibullSiteModel, Interface):
    def __init__(
        self,
        id_: ID,
        shape: AbstractParameter,
        categories: int,
        invariant: AbstractParameter = None,
        mu: AbstractParameter = None,
    ) -> None:
        super().__init__(id_, shape, categories, invariant, mu)
        self.inst = PhysherWeibullSiteModel(shape.tensor.tolist()[0], categories, None)

    def update(self, index):
        self.inst.set_shape(self._shape.tensor[index])
        if self._mu is not None:
            self.inst.set_mu(self._mu.tensor[index])
