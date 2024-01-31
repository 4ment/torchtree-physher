import numpy as np
import pytest
import torch
from torchtree import Parameter
from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_model import parse_tree

import torchtree_physher
from torchtree_physher.physher import (
    GTR,
    JC69,
    ConstantSiteModel,
    TreeLikelihoodModel,
    UnRootedTreeModel,
    WeibullSiteModel,
)


@pytest.mark.parametrize(
    "newick",
    [
        "(((A:0.01,B:0.02):0.05,C:0.03):0.0,D:0.04);",
        "(D:0.04,(C:0.03,(B:0.02,A:0.01):0.5):0.0);",
    ],
)
@pytest.mark.parametrize("use_tip_states", [True, False])
def test_unrooted(newick, use_tip_states):
    taxon_list = ["A", "B", "C", "D"]
    sequence_list = [("A", "ACTG"), ("B", "ACGT"), ("C", "ACGT"), ("D", "GCGT")]
    m = ConstantSiteModel(None)
    sm = JC69()
    tree = UnRootedTreeModel(newick, taxon_list)
    branch_lengths = np.arange(0.01, 0.06, 0.01)
    tree.set_parameters(branch_lengths)
    tlk = TreeLikelihoodModel(
        sequence_list,
        tree,
        sm,
        m,
        None,
        False,
        use_tip_states,
        False,
    )
    assert tlk.log_likelihood() == pytest.approx(-21.7658748626709)

    pm = torchtree_physher.ConstantSiteModel(None)
    psm = torchtree_physher.JC69(None)
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in taxon_list])
    tree = parse_tree(taxa, {"newick": newick})
    blens = Parameter(None, torch.tensor(branch_lengths))
    ptree = torchtree_physher.UnRootedTreeModel(None, tree, taxa, blens)
    sequences = [Sequence(taxon, seq) for taxon, seq in sequence_list]
    alignment = Alignment(None, sequences, taxa, NucleotideDataType(None))
    ptlk = torchtree_physher.TreeLikelihoodModel(None, alignment, ptree, psm, pm)
    assert ptlk() == pytest.approx(-21.7658748626709)


def test_unrooted_weibull():
    m = WeibullSiteModel(0.1, 4, None, None)
    sm = JC69()
    tree = UnRootedTreeModel("(A:0.1,(B:0.1,C:0.2):0.1);", ["A", "B", "C"])
    use_ambiguities = True
    tlk = TreeLikelihoodModel(
        [("A", "ACTG"), ("B", "ACGT"), ("C", "ACGT")],
        tree,
        sm,
        m,
        None,
        use_ambiguities,
        False,
        False,
    )
    assert tlk.log_likelihood() == pytest.approx(-14.041006507122091)
    assert len(tlk.gradient()) == 4
    tree.set_parameters([0.01, 0.5, 0.2])
    assert tlk.log_likelihood() == pytest.approx(-15.25333438550469)


def test_unrooted_weibull2():
    shape = Parameter("shape", torch.tensor([0.1]))
    m = torchtree_physher.WeibullSiteModel("weibull", shape, 4)
    sm = torchtree_physher.JC69("jc")
    tree_model_json = torchtree_physher.UnRootedTreeModel.json_factory(
        "tree",
        "(A:0.2,(B:0.1,C:0.2):0.0);",
        [0.2, 0.1, 0.2],
        {"A": "", "B": "", "C": ""},
        **{"keep_branch_lengths": True}
    )
    print(tree_model_json)
    tree_model_json["type"] = "torchtree_physher." + tree_model_json["type"]
    tree_model = torchtree_physher.UnRootedTreeModel.from_json(
        tree_model_json,
        {},
    )
    tlk = torchtree_physher.TreeLikelihoodModel(
        "id",
        [("A", "ACTG"), ("B", "ACGT"), ("C", "ACGT")],
        tree_model,
        sm,
        m,
        None,
        False,
        False,
        False,
    )
    shape.tensor = torch.tensor([[0.1]])
    tree_model._branch_lengths.tensor = torch.tensor([[0.2, 0.1, 0.2]])
    m.update(0)
    tree_model.update(0)
    # tree_model.inst.set_parameters([0.01, 0.5, 0.2])
    # m.inst.set_parameters([0.1])
    # tlk.lp_needs_update = True
    assert tlk() == pytest.approx(-14.041006507122091)
    tree_model._branch_lengths.tensor = torch.tensor([[0.01, 0.5, 0.2]])
    m.update(0)
    assert tlk() == pytest.approx(-15.25333438550469)

    shape.tensor = torch.tensor([[0.5]])
    tree_model._branch_lengths.tensor = torch.tensor([[0.2, 0.1, 0.2]])
    m.update(0)
    tree_model.update(0)
    assert tlk() == pytest.approx(-13.26174426466766)


@pytest.mark.parametrize(
    "newick",
    [
        "(((A:0.01,B:0.02):0.05,C:0.03):0.0,D:0.04);",
        "(D:0.04,(C:0.03,(B:0.02,A:0.01):0.5):0.0);",
    ],
)
@pytest.mark.parametrize("use_tip_states", [True, False])
def test_unrooted_GTR(newick, use_tip_states):
    taxon_list = ["A", "B", "C", "D"]
    sequence_list = [("A", "ACTG"), ("B", "ACGT"), ("C", "ACGT"), ("D", "GCGT")]
    m = ConstantSiteModel(None)
    sm = GTR([1.0 / 6] * 6, [0.25] * 4)
    tree = UnRootedTreeModel(newick, taxon_list)
    branch_lengths = np.arange(0.01, 0.06, 0.01)
    tree.set_parameters(branch_lengths)
    tlk = TreeLikelihoodModel(
        sequence_list,
        tree,
        sm,
        m,
        None,
        False,
        use_tip_states,
        False,
    )
    assert tlk.log_likelihood() == pytest.approx(-21.7658748626709)

    pm = torchtree_physher.ConstantSiteModel(None)
    psm = torchtree_physher.GTR(
        None,
        Parameter("rates", torch.full([6], 1.0 / 6)),
        Parameter("freqs", torch.full([4], 0.25)),
    )
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in taxon_list])
    tree = parse_tree(taxa, {"newick": newick})
    blens = Parameter(None, torch.tensor(branch_lengths))
    ptree = torchtree_physher.UnRootedTreeModel(None, tree, taxa, blens)
    sequences = [Sequence(taxon, seq) for taxon, seq in sequence_list]
    alignment = Alignment(None, sequences, taxa, NucleotideDataType(None))
    psm._rates.tensor = torch.full([1, 6], 1.0 / 6)
    psm._frequencies.tensor = torch.full([1, 4], 0.25)
    ptree._branch_lengths.tensor = torch.tensor(branch_lengths).unsqueeze(0)
    ptlk = torchtree_physher.TreeLikelihoodModel(None, alignment, ptree, psm, pm)
    assert ptlk() == pytest.approx(-21.7658748626709)

    psm._rates.tensor = torch.full([1, 6], 1.0 / 6)
    psm._frequencies.tensor = torch.full([1, 4], 0.25)
    ptree._branch_lengths.tensor = torch.tensor(branch_lengths).unsqueeze(0) + 1.0
    ptlk = torchtree_physher.TreeLikelihoodModel(None, alignment, ptree, psm, pm)
    assert ptlk() != pytest.approx(-21.7658748626709)
