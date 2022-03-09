import torch
import torchtree.distributions.ctmc_scale
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.typing import ID

from physherpy.interface import Interface
from physherpy.physher import CTMCScaleModel as PhysherCTMCScaleModel


class CTMCScale(torchtree.distributions.ctmc_scale.CTMCScale, Interface):
    def __init__(self, id_: ID, x: AbstractParameter, tree_model) -> None:
        super().__init__(id_, x, tree_model)
        self.inst = PhysherCTMCScaleModel(x.tensor.tolist(), tree_model.inst)

    def update(self, index):
        self.inst.set_parameters(self.x.tensor[index].detach().numpy())

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return CTMCScaleAutogradFunction.apply(
            self.inst,
            [self, self.tree_model],
            self.tree_model._internal_heights.tensor,
            self.x.tensor,
        )


class CTMCScaleAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inst,
        models,
        internal_heights,
        rates,
    ):
        ctx.inst = inst
        ctx.save_for_backward(rates, internal_heights)

        log_probs = []
        grads = []
        for i in range(rates.shape[0]):
            for model in models:
                model.update(i)
            log_probs.append(torch.tensor([inst.log_likelihood()]))
            if rates.requires_grad:
                grads.append(torch.tensor(inst.gradient()))
        ctx.grads = torch.stack(grads) if rates.requires_grad else None
        return torch.stack(log_probs)

    @staticmethod
    def backward(ctx, grad_output):
        (
            internal_heights,
            rates,
        ) = ctx.saved_tensors
        internal_heights_grad = (
            ctx.grads[..., internal_heights.shape[-1] :] * grad_output
        )
        rates_grad = ctx.grads[..., : internal_heights.shape[-1]] * grad_output
        return (
            None,  # inst
            None,  # model
            internal_heights_grad,
            rates_grad,
        )
