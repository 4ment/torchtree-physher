import operator
from functools import reduce

import torch
import torchtree.evolution.coalescent
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.typing import ID

import torchtree_physher.physher.gradient_flags as flags
from torchtree_physher.interface import Interface
from torchtree_physher.physher import (
    ConstantCoalescentModel as PhysherConstantCoalescentModel,
)
from torchtree_physher.physher import (
    PiecewiseConstantCoalescentGridModel as PhysherPiecewiseConstantCoalescentGridModel,
)
from torchtree_physher.physher import (
    PiecewiseConstantCoalescentModel as PhysherPiecewiseConstantCoalescentModel,
)
from torchtree_physher.physher import (
    PiecewiseLinearCoalescentGridModel as PhysherPiecewiseLinearCoalescentGridModel,
)
from torchtree_physher.physher import (
    ReparameterizedTimeTreeModel as PhysherReparameterizedTimeTreeModel,
)

from .utils import flatten_2D


def evaluate_coalescent(
    inst: PhysherConstantCoalescentModel,
    coalescent,
    tree_model: PhysherReparameterizedTimeTreeModel,
):
    fn = CoalescentAutogradFunction.apply
    return fn(
        inst,
        [coalescent, tree_model],
        coalescent.theta.tensor,
        tree_model._internal_heights.tensor,
    )


class ConstantCoalescentModel(
    torchtree.evolution.coalescent.ConstantCoalescentModel, Interface
):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        tree_model,
        **kwargs,
    ) -> None:
        super().__init__(id_, theta, tree_model)
        self.inst = PhysherConstantCoalescentModel(
            flatten_2D(theta.tensor)[0].item(), tree_model.inst
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        tensor_flatten = flatten_2D(self.theta.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())


class PiecewiseConstantCoalescentModel(
    torchtree.evolution.coalescent.PiecewiseConstantCoalescentModel, Interface
):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        tree_model,
    ) -> None:
        super().__init__(id_, theta, tree_model)
        self.inst = PhysherPiecewiseConstantCoalescentModel(
            flatten_2D(theta.tensor)[0].tolist(), tree_model.inst
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        tensor_flatten = flatten_2D(self.theta.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())


class PiecewiseConstantCoalescentGridModel(
    torchtree.evolution.coalescent.PiecewiseConstantCoalescentGridModel, Interface
):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        grid: AbstractParameter,
        tree_model,
    ) -> None:
        super().__init__(id_, theta, grid, tree_model)
        self.inst = PhysherPiecewiseConstantCoalescentGridModel(
            flatten_2D(theta.tensor)[0].tolist(),
            tree_model.inst,
            grid.tensor[-1].item(),
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        tensor_flatten = flatten_2D(self.theta.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())


class PiecewiseLinearCoalescentGridModel(
    torchtree.evolution.coalescent.PiecewiseLinearCoalescentGridModel, Interface
):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        grid: AbstractParameter,
        tree_model,
    ) -> None:
        super().__init__(id_, theta, grid, tree_model)
        self.inst = PhysherPiecewiseLinearCoalescentGridModel(
            flatten_2D(theta.tensor)[0].tolist(),
            tree_model.inst,
            grid.tensor[-1].item(),
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        tensor_flatten = flatten_2D(self.theta.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())


class CoalescentAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inst,
        models,
        thetas,
        internal_heights,
    ):
        ctx.inst = inst
        ctx.save_for_backward(thetas, internal_heights)

        log_probs = []
        grads = []
        options = {"dtype": internal_heights.dtype, "device": internal_heights.device}
        dim = reduce(operator.mul, thetas.shape[:-1], 1)
        requires_grad = False

        physher_flags = []
        if thetas.requires_grad:
            physher_flags.append(flags.THETA)
        if internal_heights.requires_grad:
            physher_flags.append(flags.TREE_RATIO)
        if len(physher_flags) > 0:
            inst.request_gradient(physher_flags)
            requires_grad = True

        for i in range(dim):
            for model in models:
                model.update(i)
            log_probs.append(inst.log_likelihood())
            if requires_grad:
                grads.append(torch.tensor(inst.gradient(), **options))
        ctx.grads = torch.stack(grads) if len(grads) > 0 else None
        return torch.tensor(log_probs, **options).view(thetas.shape[:-1] + (1,))

    @staticmethod
    def backward(ctx, grad_output):
        (
            thetas,
            internal_heights,
        ) = ctx.saved_tensors
        offset = 0
        if thetas.requires_grad:
            thetas_grad = ctx.grads[..., : thetas.shape[-1]] * grad_output
            offset += thetas.shape[-1]
        else:
            thetas_grad = None

        if internal_heights.requires_grad:
            internal_heights_grad = ctx.grads[..., offset:] * grad_output
        else:
            internal_heights_grad = None

        return (
            None,  # inst
            None,  # models
            thetas_grad,
            internal_heights_grad,
        )
