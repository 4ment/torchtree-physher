import torch
import torchtree
from torch.distributions import Transform
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.taxa import Taxa
from torchtree.evolution.tree_model import (
    ReparameterizedTimeTreeModel as TReparameterizedTimeTreeModel,
)
from torchtree.evolution.tree_model import UnRootedTreeModel as TUnRootedTreeModel
from torchtree.typing import ID

from physherpy.interface import Interface
from physherpy.physher import (
    ReparameterizedTimeTreeModel as PhysherReparameterizedTimeTreeModel,
)
from physherpy.physher import UnRootedTreeModel as PhysherUnRootedTreeModel


def flatten_2D(tensor: torch.tensor) -> torch.tensor:
    """Flatten batch dimensions.

    :param tensor: tensor of any dimension (can be None)
    :return: tensor with batch dimensions flatten

    Example:

        >>> t = torch.rand((2, 3, 4))
        >>> t2 = flatten_2D(t)
        >>> t2.shape
        torch.Size([6, 4])
        >>> torch.all(t2.view((2, 3, 4)) == t)
        tensor(True)
        >>> flatten_2D(None) is None
        True
    """
    if tensor is None or len(tensor.shape) == 2:
        tensor_flatten = tensor
    elif len(tensor.shape) > 2:
        tensor_flatten = torch.flatten(tensor, end_dim=-2)
    else:
        tensor_flatten = tensor.unsqueeze(0)
    return tensor_flatten


class GeneralNodeHeightTransform(Transform):
    r"""
    Transform from ratios to node heights.
    """

    def __init__(self, tree, inst, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.inst = inst

    def _call(self, x):
        fn = NodeHeightAutogradFunction.apply
        return fn(self.inst, x)

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(1)


class NodeHeightAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inst, ratios_root_height):
        ctx.inst = inst
        node_heights = []

        tensor_flatten = flatten_2D(ratios_root_height)

        params_numpy = tensor_flatten.detach().numpy()
        for batch_idx in range(tensor_flatten.shape[0]):
            inst.set_parameters(params_numpy[batch_idx, ...])
            node_heights.append(torch.tensor(inst.get_node_heights()))

        node_heights = torch.stack(node_heights)
        if len(ratios_root_height.shape) != node_heights.shape:
            node_heights = node_heights.view(ratios_root_height.shape)
        return node_heights

    @staticmethod
    def backward(ctx, grad_output):
        grad = []
        tensor_flatten = flatten_2D(grad_output)
        grad_output_numpy = tensor_flatten.numpy()
        for batch_idx in range(tensor_flatten.shape[0]):
            grad.append(
                torch.tensor(
                    ctx.inst.gradient_transform_jvp(grad_output_numpy[batch_idx, ...])
                )
            )
        grad = torch.stack(grad)
        if grad.shape != grad_output.shape:
            grad = grad.view(grad_output.shape)
        return None, grad


class UnRootedTreeModel(TUnRootedTreeModel, Interface):
    def __init__(
        self, id_: ID, tree, taxa: Taxa, branch_lengths: AbstractParameter
    ) -> None:
        super().__init__(id_, tree, taxa, branch_lengths)
        taxon_list = [taxon.id for taxon in taxa]
        self.inst = PhysherUnRootedTreeModel(
            tree.as_string('newick').replace("'", "").replace('[&R] ', ''), taxon_list
        )

    def update(self, index):
        self.inst.set_parameters(self._branch_lengths.tensor[index].detach().numpy())


class ReparameterizedTimeTreeModel(TReparameterizedTimeTreeModel, Interface):
    def __init__(
        self, id_: ID, tree, taxa: Taxa, ratios_root_heights: AbstractParameter
    ) -> None:
        super().__init__(id_, tree, taxa, ratios_root_heights)
        taxon_list = [taxon.id for taxon in taxa]

        # physher needs dates in the right order
        l = []
        for node in tree.postorder_node_iter():
            if node.is_leaf():
                l.append(node.taxon.label)
        dates = [self.sampling_times.tolist()[taxon_list.index(taxon)] for taxon in l]

        self.inst = PhysherReparameterizedTimeTreeModel(
            tree.as_string('newick').replace("'", "").replace('[&R] ', ''),
            taxon_list,
            dates,
        )
        self.inst.set_parameters(ratios_root_heights.tensor.tolist())

    def update(self, index):
        self.inst.set_parameters(self._internal_heights.tensor[index].detach().numpy())

    @property
    def node_heights(self) -> torch.Tensor:
        if self.heights_need_update:
            self._heights = GeneralNodeHeightTransform(self, self.inst)(
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
