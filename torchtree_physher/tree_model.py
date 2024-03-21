import torch
from torch.distributions import Transform
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.taxa import Taxa
from torchtree.evolution.tree_model import (
    ReparameterizedTimeTreeModel as TReparameterizedTimeTreeModel,
)
from torchtree.evolution.tree_model import TimeTreeModel as TTimeTreeModel
from torchtree.evolution.tree_model import UnRootedTreeModel as TUnRootedTreeModel
from torchtree.typing import ID

from torchtree_physher.interface import Interface
from torchtree_physher.physher import (
    ReparameterizedTimeTreeModel as PhysherReparameterizedTimeTreeModel,
)
from torchtree_physher.physher import TimeTreeModel as PhysherTimeTreeModel
from torchtree_physher.physher import UnRootedTreeModel as PhysherUnRootedTreeModel
from torchtree_physher.physher import tree_transform_flags
from torchtree_physher.utils import flatten_2D


class GeneralNodeHeightTransform(Transform):
    r"""
    Transform from ratios to node heights.
    """

    def __init__(self, inst, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.inst = inst

    def _call(self, x):
        fn = NodeHeightAutogradFunction.apply
        return fn(self.inst, x)

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        fn = NodeHeightJacobianAutogradFunction.apply
        return fn(self.inst, x)


class NodeHeightAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inst, ratios_root_height):
        ctx.inst = inst
        node_heights = []

        tensor_flatten = flatten_2D(ratios_root_height)

        params_numpy = tensor_flatten.detach().numpy()
        for batch_idx in range(tensor_flatten.shape[0]):
            inst.set_parameters(params_numpy[batch_idx, ...])
            node_heights.append(torch.tensor(inst.node_heights()))

        node_heights = torch.stack(node_heights)
        if ratios_root_height.shape != node_heights.shape:
            node_heights = node_heights.view(ratios_root_height.shape)
        if ratios_root_height.requires_grad:
            ctx.save_for_backward(node_heights)
        return node_heights

    @staticmethod
    def backward(ctx, grad_output):
        (heights,) = ctx.saved_tensors
        grad = []
        tensor_flatten = flatten_2D(grad_output)
        heights_numpy = flatten_2D(heights).numpy()
        grad_output_numpy = tensor_flatten.numpy()
        for batch_idx in range(tensor_flatten.shape[0]):
            grad.append(
                torch.tensor(
                    ctx.inst.gradient_transform_jvp(
                        grad_output_numpy[batch_idx, ...],
                        heights_numpy[batch_idx, ...],
                    )
                )
            )
        grad = torch.stack(grad)
        if grad.shape != grad_output.shape:
            grad = grad.view(grad_output.shape)
        return None, grad


class NodeHeightJacobianAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inst, ratios_root_height):
        ctx.inst = inst
        ctx.save_for_backward(ratios_root_height)

        log_probs = []
        grads = []
        options = {
            "dtype": ratios_root_height.dtype,
            "device": ratios_root_height.device,
        }

        tensor_flatten = flatten_2D(ratios_root_height)
        params_numpy = tensor_flatten.detach().numpy()
        for batch_idx in range(tensor_flatten.shape[0]):
            inst.set_parameters(params_numpy[batch_idx, ...])
            log_probs.append(inst.transform_jacobian())
            if ratios_root_height.requires_grad:
                grads.append(
                    torch.tensor(inst.gradient_transform_jacobian(), **options)
                )

        ctx.grads = None
        if ratios_root_height.requires_grad:
            ctx.grads = torch.stack(grads).view(ratios_root_height.shape)

        return torch.tensor(log_probs, **options).view(
            ratios_root_height.shape[:-1] + (1,)
        )

    @staticmethod
    def backward(ctx, grad_output):
        return None, ctx.grads * grad_output


class UnRootedTreeModel(TUnRootedTreeModel, Interface):
    def __init__(
        self, id_: ID, tree, taxa: Taxa, branch_lengths: AbstractParameter
    ) -> None:
        super().__init__(id_, tree, taxa, branch_lengths)
        taxon_list = [taxon.id for taxon in taxa]
        self.inst = PhysherUnRootedTreeModel(
            tree.as_string("newick").replace("'", "").replace("[&R] ", ""), taxon_list
        )

    def update(self, index):
        tensor_flatten = flatten_2D(self._branch_lengths.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())


class ReparameterizedTimeTreeModel(TReparameterizedTimeTreeModel, Interface):
    def __init__(
        self,
        id_: ID,
        tree,
        taxa: Taxa,
        ratios_root_height: AbstractParameter = None,
        shifts: AbstractParameter = None,
    ) -> None:
        if ratios_root_height is not None:
            super().__init__(id_, tree, taxa, ratios_root_height=ratios_root_height)
            parameters = ratios_root_height
            self.zero_jacobian = False
            transform_flag = tree_transform_flags.RATIO
        else:
            parameters = shifts
            super().__init__(id_, tree, taxa, shifts=shifts)
            # the log det Jacobian is zero
            self.zero_jacobian = True
            transform_flag = tree_transform_flags.SHIFT

        taxon_list = [taxon.id for taxon in taxa]

        self.inst = PhysherReparameterizedTimeTreeModel(
            tree.as_string("newick").replace("'", "").replace("[&R] ", ""),
            taxon_list,
            self.sampling_times.tolist(),
            transform_flag,
        )
        self.inst.set_parameters(parameters.tensor.tolist())

    def update(self, index):
        tensor_flatten = flatten_2D(self._internal_heights.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())

    def _call(self, *args, **kwargs) -> torch.Tensor:
        if self.zero_jacobian:
            return torch.zeros(
                1,
                dtype=self._internal_heights.dtype,
                device=self._internal_heights.device,
            )

        if self.heights_need_update:
            self.update_node_heights()
        return GeneralNodeHeightTransform(self.inst).log_abs_det_jacobian(
            self._internal_heights.tensor, self._heights
        )

    @property
    def node_heights(self) -> torch.Tensor:
        if self.heights_need_update:
            self._heights = GeneralNodeHeightTransform(self.inst)(
                self._internal_heights.tensor
            )
            self._node_heights = torch.cat(
                (
                    self.sampling_times.expand(
                        self._internal_heights.tensor.shape[:-1] + (-1,)
                    ),
                    self._heights,
                ),
                -1,
            )
            self.heights_need_update = False
        return self._node_heights


class TimeTreeModel(TTimeTreeModel, Interface):
    def __init__(
        self,
        id_: ID,
        tree,
        taxa: Taxa,
        internal_heights: AbstractParameter,
    ) -> None:
        super().__init__(id_, tree, taxa, internal_heights)
        taxon_list = [taxon.id for taxon in taxa]

        self.inst = PhysherTimeTreeModel(
            tree.as_string("newick").replace("'", "").replace("[&R] ", ""),
            taxon_list,
            self.sampling_times.tolist(),
        )
        self.inst.set_parameters(internal_heights.tensor.tolist())

    def update(self, index):
        tensor_flatten = flatten_2D(self._internal_heights.tensor)
        self.inst.set_parameters(tensor_flatten[index].detach().numpy())
