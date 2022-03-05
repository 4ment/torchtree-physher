import torch

import torchtree.evolution.coalescent
from physherpy.interface import Interface
from physherpy.physher import ConstantCoalescentModel as PhysherConstantCoalescentModel
from physherpy.physher import (
    PiecewiseConstantCoalescentGridModel as PhysherPiecewiseConstantCoalescentGridModel,
)
from physherpy.physher import (
    PiecewiseConstantCoalescentModel as PhysherPiecewiseConstantCoalescentModel,
)
from physherpy.physher import (
    ReparameterizedTimeTreeModel as PhysherReparameterizedTimeTreeModel,
)
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.typing import ID


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
    ) -> None:
        super().__init__(id_, theta, tree_model)
        self.inst = PhysherConstantCoalescentModel(theta.tensor.item(), tree_model.inst)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        self.inst.set_parameters(self.theta.tensor[index].detach().numpy())


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
            theta.tensor.tolist(), tree_model.inst
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        self.inst.set_parameters(self.theta.tensor[index].detach().numpy())


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
            theta.tensor.tolist(), tree_model.inst, grid.tensor[-1].item()
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return evaluate_coalescent(self.inst, self, self.tree_model)

    def update(self, index):
        self.inst.set_parameters(self.theta.tensor[index].detach().numpy())


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
        for i in range(thetas.shape[0]):
            for model in models:
                model.update(i)
            log_probs.append(torch.tensor([inst.log_likelihood()]))
            if thetas.requires_grad:
                grads.append(torch.tensor(inst.gradient()))
        ctx.grads = torch.stack(grads) if thetas.requires_grad else None
        return torch.stack(log_probs)

    @staticmethod
    def backward(ctx, grad_output):
        (
            thetas,
            internal_heights,
        ) = ctx.saved_tensors
        thetas_grad = ctx.grads[..., : thetas.shape[-1]] * grad_output
        internal_heights_grad = ctx.grads[..., thetas.shape[-1] :] * grad_output
        return (
            None,  # inst
            None,  # models
            thetas_grad,
            internal_heights_grad,
        )
