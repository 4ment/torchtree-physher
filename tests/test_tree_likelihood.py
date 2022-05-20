import numpy as np
import pytest
import torch
from torchtree import Parameter
from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_model import parse_tree

import physherpy.physher


@pytest.mark.parametrize(
    "newick",
    [
        "(((A:0.01,B:0.02):0.05,C:0.03):0.0,D:0.04);",
        "(D:0.04,(C:0.03,(B:0.02,A:0.01):0.5):0.0);",
    ],
)
@pytest.mark.parametrize('use_tip_states', [True, False])
def test_unrooted(newick, use_tip_states):
    taxon_list = ['A', 'B', 'C', 'D']
    sequence_list = [('A', 'ACTG'), ('B', 'ACGT'), ('C', 'ACGT'), ('D', 'GCGT')]
    m = physherpy.physher.ConstantSiteModel(None)
    sm = physherpy.physher.JC69()
    tree = physherpy.physher.UnRootedTreeModel(newick, taxon_list)
    branch_lengths = np.arange(0.01, 0.06, 0.01)
    tree.set_parameters(branch_lengths)
    tlk = physherpy.physher.TreeLikelihoodModel(
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

    pm = physherpy.ConstantSiteModel(None)
    psm = physherpy.JC69(None)
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in taxon_list])
    tree = parse_tree(taxa, {'newick': newick})
    blens = Parameter(None, torch.tensor(branch_lengths))
    ptree = physherpy.UnRootedTreeModel(None, tree, taxa, blens)
    sequences = [Sequence(taxon, seq) for taxon, seq in sequence_list]
    alignment = Alignment(None, sequences, taxa, NucleotideDataType(None))
    ptlk = physherpy.TreeLikelihoodModel(None, alignment, ptree, psm, pm)
    assert ptlk() == pytest.approx(-21.7658748626709)


def test_unrooted_weibull():
    m = physherpy.physher.WeibullSiteModel(0.1, 4, None)
    sm = physherpy.physher.JC69()
    tree = physherpy.physher.UnRootedTreeModel(
        "(A:0.1,(B:0.1,C:0.2):0.1);", ['A', 'B', 'C']
    )
    use_ambiguities = True
    tlk = physherpy.physher.TreeLikelihoodModel(
        [('A', 'ACTG'), ('B', 'ACGT'), ('C', 'ACGT')],
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
    shape = Parameter('shape', torch.tensor([0.1]))
    m = physherpy.site_model.WeibullSiteModel('weibull', shape, 4)
    sm = physherpy.JC69('jc')
    tree_model_json = physherpy.UnRootedTreeModel.json_factory(
        'tree',
        '(A:0.2,(B:0.1,C:0.2):0.0);',
        [0.2, 0.1, 0.2],
        {'A': '', 'B': '', 'C': ''},
        **{'keep_branch_lengths': True}
    )
    print(tree_model_json)
    tree_model_json['type'] = 'physherpy.' + tree_model_json['type']
    tree_model = physherpy.UnRootedTreeModel.from_json(
        tree_model_json,
        {},
    )
    tlk = physherpy.TreeLikelihoodModel(
        'id',
        [('A', 'ACTG'), ('B', 'ACGT'), ('C', 'ACGT')],
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
@pytest.mark.parametrize('use_tip_states', [True, False])
def test_unrooted_GTR(newick, use_tip_states):
    taxon_list = ['A', 'B', 'C', 'D']
    sequence_list = [('A', 'ACTG'), ('B', 'ACGT'), ('C', 'ACGT'), ('D', 'GCGT')]
    m = physherpy.physher.ConstantSiteModel(None)
    sm = physherpy.physher.GTR([1.0 / 6] * 6, [0.25] * 4)
    tree = physherpy.physher.UnRootedTreeModel(newick, taxon_list)
    branch_lengths = np.arange(0.01, 0.06, 0.01)
    tree.set_parameters(branch_lengths)
    tlk = physherpy.physher.TreeLikelihoodModel(
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

    pm = physherpy.ConstantSiteModel(None)
    psm = physherpy.GTR(
        None,
        Parameter('rates', torch.full([6], 1.0 / 6)),
        Parameter('freqs', torch.full([4], 0.25)),
    )
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in taxon_list])
    tree = parse_tree(taxa, {'newick': newick})
    blens = Parameter(None, torch.tensor(branch_lengths))
    ptree = physherpy.UnRootedTreeModel(None, tree, taxa, blens)
    sequences = [Sequence(taxon, seq) for taxon, seq in sequence_list]
    alignment = Alignment(None, sequences, taxa, NucleotideDataType(None))
    psm._rates.tensor = torch.full([1, 6], 1.0 / 6)
    psm._frequencies.tensor = torch.full([1, 4], 0.25)
    ptree._branch_lengths.tensor = torch.tensor(branch_lengths).unsqueeze(0)
    ptlk = physherpy.TreeLikelihoodModel(None, alignment, ptree, psm, pm)
    assert ptlk() == pytest.approx(-21.7658748626709)

    psm._rates.tensor = torch.full([1, 6], 1.0 / 6)
    psm._frequencies.tensor = torch.full([1, 4], 0.25)
    ptree._branch_lengths.tensor = torch.tensor(branch_lengths).unsqueeze(0) + 1.0
    ptlk = physherpy.TreeLikelihoodModel(None, alignment, ptree, psm, pm)
    assert ptlk() != pytest.approx(-21.7658748626709)
