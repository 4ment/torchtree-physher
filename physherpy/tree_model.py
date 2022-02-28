from physherpy.interface import Interface
from physherpy.physher import TreeModel as PhysherTreeModel
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.evolution.taxa import Taxa
from torchtree.evolution.tree_model import UnRootedTreeModel as TUnRootedTreeModel
from torchtree.typing import ID


class UnRootedTreeModel(TUnRootedTreeModel, Interface):
    def __init__(
        self, id_: ID, tree, taxa: Taxa, branch_lengths: AbstractParameter
    ) -> None:
        super().__init__(id_, tree, taxa, branch_lengths)
        taxon_list = [taxon.id for taxon in taxa]
        self.inst = PhysherTreeModel(
            tree.as_string('newick').replace("'", "").replace('[&R] ', ''),
            taxon_list,
            None,
        )

    def update(self, index):
        self.inst.set_parameters(self._branch_lengths.tensor[index].tolist())
