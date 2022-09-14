import torch
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.utils import process_object
from torchtree.evolution.site_model import (
    ConstantSiteModel as TorchtreeConstantSiteModel,
)
from torchtree.evolution.site_model import (
    InvariantSiteModel as TorchtreeInvariantSiteModel,
)
from torchtree.evolution.site_model import UnivariateDiscretizedSiteModel
from torchtree.evolution.site_model import WeibullSiteModel as TorchtreeWeibullSiteModel
from torchtree.typing import ID

from .interface import Interface
from .physher import ConstantSiteModel as PhysherConstantSiteModel
from .physher import GammaSiteModel as PhysherGammaSiteModel
from .physher import InvariantSiteModel as PhysherInvariantSiteModel
from .physher import WeibullSiteModel as PhysherWeibullSiteModel


class ConstantSiteModel(TorchtreeConstantSiteModel, Interface):
    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_, mu)
        mu_float = None if mu is None else mu.tensor.item()
        self.inst = PhysherConstantSiteModel(mu_float)

    def update(self, index):
        if self._mu is not None:
            self.inst.set_mu(self._mu.tensor[index])


class InvariantSiteModel(TorchtreeInvariantSiteModel, Interface):
    def __init__(
        self, id_: ID, invariant: AbstractParameter, mu: AbstractParameter = None
    ) -> None:
        super().__init__(id_, invariant, mu)
        mu_float = None if mu is None else mu.tensor.item()
        self.inst = PhysherInvariantSiteModel(invariant.tensor.item(), mu_float)

    def update(self, index):
        self.inst.set_proportion_invariant(self.invariant[index].item())
        if self._mu is not None:
            self.inst.set_mu(self._mu.tensor[index].item())


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
        mu_float = None if mu is None else mu.tensor.item()
        invariant_float = None if invariant is None else invariant.tensor.item()
        self.inst = PhysherWeibullSiteModel(
            shape.tensor.item(), categories, invariant_float, mu_float
        )

    def update(self, index):
        self.inst.set_shape(self.shape[index].item())
        if self._mu is not None:
            self.inst.set_mu(self._mu.tensor[index].item())
        if self._invariant is not None:
            self.inst.set_proportion_invariant(self._invariant.tensor[index].item())


class GammaSiteModel(UnivariateDiscretizedSiteModel, Interface):
    def __init__(
        self,
        id_: ID,
        shape: AbstractParameter,
        categories: int,
        invariant: AbstractParameter = None,
        mu: AbstractParameter = None,
    ) -> None:
        super().__init__(id_, shape, categories, invariant, mu)
        mu_float = None if mu is None else mu.tensor.item()
        invariant_float = None if invariant is None else invariant.tensor.item()
        self.inst = PhysherGammaSiteModel(
            shape.tensor.item(), categories, invariant_float, mu_float
        )

    @property
    def shape(self) -> torch.Tensor:
        return self._parameter.tensor

    def update(self, index) -> None:
        self.inst.set_shape(self.shape[index].item())
        if self._mu is not None:
            self.inst.set_mu(self._mu.tensor[index].item())
        if self._invariant is not None:
            self.inst.set_proportion_invariant(self._invariant.tensor[index].item())

    def inverse_cdf(
        self, parameter: torch.Tensor, quantile: torch.Tensor, invariant: torch.Tensor
    ) -> torch.Tensor:
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        shape = process_object(data['shape'], dic)
        categories = data['categories']
        invariant = None
        if 'invariant' in data:
            invariant = process_object(data['invariant'], dic)
        if 'mu' in data:
            mu = process_object(data['mu'], dic)
        else:
            mu = None
        return cls(id_, shape, categories, invariant, mu)
