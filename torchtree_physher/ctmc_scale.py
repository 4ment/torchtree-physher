import torch
import torchtree.distributions.ctmc_scale
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.typing import ID

import torchtree_physher.physher.gradient_flags as flags
from torchtree_physher.interface import Interface
from torchtree_physher.physher import CTMCScaleModel as PhysherCTMCScaleModel


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

        ctmc_model, tree_model = models

        # the rates can be fixed so its shape can be different from internal_heights
        # depending on the number of samples. For a strict clock with n internal nodes
        # we can have: rates.shape==[1] and internal_heights.shape==[100,n]
        if rates.dim() == 1:
            ctmc_model.update(0)
            if internal_heights.requires_grad and not rates.requires_grad:
                inst.request_gradient([flags.TREE_RATIO])

        # MAP or HMC
        if internal_heights.dim() == 1:
            internal_heights = internal_heights.unsqueeze(0)

        log_probs = []
        grads = []
        for i in range(internal_heights.shape[0]):
            if rates.dim() > 1:
                ctmc_model.update(i)
            tree_model.update(i)

            log_probs.append(torch.tensor([inst.log_likelihood()]))
            if internal_heights.requires_grad:
                grads.append(torch.tensor(inst.gradient()))
        ctx.grads = torch.stack(grads) if internal_heights.requires_grad else None
        return torch.stack(log_probs)

    @staticmethod
    def backward(ctx, grad_output):
        (
            rates,
            internal_heights,
        ) = ctx.saved_tensors
        internal_heights_grad = (
            ctx.grads[..., : internal_heights.shape[-1]] * grad_output
        )
        if rates.requires_grad:
            rates_grad = ctx.grads[..., internal_heights.shape[-1] :] * grad_output
        else:
            rates_grad = None

        return (
            None,  # inst
            None,  # model
            internal_heights_grad,
            rates_grad,
        )
