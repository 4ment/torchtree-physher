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

from torchtree_physher.interface import Interface
from torchtree_physher.physher import ConstantSiteModel as PhysherConstantSiteModel
from torchtree_physher.physher import GammaSiteModel as PhysherGammaSiteModel
from torchtree_physher.physher import InvariantSiteModel as PhysherInvariantSiteModel
from torchtree_physher.physher import WeibullSiteModel as PhysherWeibullSiteModel
from torchtree_physher.utils import flatten_2D


class ConstantSiteModel(TorchtreeConstantSiteModel, Interface):
    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_, mu)
        mu_float = None if mu is None else flatten_2D(mu.tensor)[0].item()
        self.inst = PhysherConstantSiteModel(mu_float)

    def update(self, index):
        if self._mu is not None:
            self.inst.set_mu(flatten_2D(self._mu.tensor)[index])


class InvariantSiteModel(TorchtreeInvariantSiteModel, Interface):
    def __init__(
        self, id_: ID, invariant: AbstractParameter, mu: AbstractParameter = None
    ) -> None:
        super().__init__(id_, invariant, mu)
        mu_float = None if mu is None else flatten_2D(mu.tensor)[0].item()
        self.inst = PhysherInvariantSiteModel(
            flatten_2D(invariant.tensor)[0].item(), mu_float
        )

    def update(self, index):
        self.inst.set_proportion_invariant(flatten_2D(self.invariant)[index].item())
        if self._mu is not None:
            self.inst.set_mu(flatten_2D(self._mu.tensor)[index].item())


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
        mu_float = None if mu is None else flatten_2D(mu.tensor)[0].item()
        invariant_float = (
            None if invariant is None else flatten_2D(invariant.tensor)[0].item()
        )
        self.inst = PhysherWeibullSiteModel(
            flatten_2D(shape.tensor)[0].item(), categories, invariant_float, mu_float
        )

    def update(self, index):
        self.inst.set_shape(flatten_2D(self.shape)[index].item())
        if self._mu is not None:
            self.inst.set_mu(flatten_2D(self._mu.tensor)[index].item())
        if self._invariant is not None:
            self.inst.set_proportion_invariant(
                flatten_2D(self._invariant.tensor)[index].item()
            )


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
        mu_float = None if mu is None else flatten_2D(mu.tensor)[0].item()
        invariant_float = (
            None if invariant is None else flatten_2D(invariant.tensor)[0].item()
        )
        self.inst = PhysherGammaSiteModel(
            flatten_2D(shape.tensor)[0].item(), categories, invariant_float, mu_float
        )

    @property
    def shape(self) -> torch.Tensor:
        return self._parameter.tensor

    def update(self, index) -> None:
        self.inst.set_shape(flatten_2D(self.shape)[index].item())
        if self._mu is not None:
            self.inst.set_mu(flatten_2D(self._mu.tensor)[index].item())
        if self._invariant is not None:
            self.inst.set_proportion_invariant(
                flatten_2D(self._invariant.tensor)[index].item()
            )

    def inverse_cdf(
        self, parameter: torch.Tensor, quantile: torch.Tensor, invariant: torch.Tensor
    ) -> torch.Tensor:
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        shape = process_object(data["shape"], dic)
        categories = data["categories"]
        invariant = None
        if "invariant" in data:
            invariant = process_object(data["invariant"], dic)
        if "mu" in data:
            mu = process_object(data["mu"], dic)
        else:
            mu = None
        return cls(id_, shape, categories, invariant, mu)
