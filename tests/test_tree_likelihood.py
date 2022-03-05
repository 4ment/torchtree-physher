import numpy as np
import pytest

import physherpy.physher


@pytest.mark.parametrize(
    "newick,branch_lengths",
    [
        ("(((A:0.01,B:0.02):0.05,C:0.03):0.0,D:0.04);", np.arange(0.01, 0.06, 0.01)),
        (
            "(D:0.04,(C:0.03,(B:0.02,A:0.01):0.5):0.0);",
            np.array([0.01, 0.02, 0.04, 0.03, 0.05]),
        ),
    ],
)
@pytest.mark.parametrize('use_ambiguities', [True, False])
def test_unrooted(newick, branch_lengths, use_ambiguities):
    m = physherpy.physher.ConstantSiteModel(None)
    sm = physherpy.physher.JC69()
    tree = physherpy.physher.UnRootedTreeModel(newick, ['A', 'B', 'C', 'D'])
    tree.set_parameters(np.arange(0.01, 0.06, 0.01))
    tlk = physherpy.physher.TreeLikelihoodModel(
        [('A', 'ACTG'), ('B', 'ACGT'), ('C', 'ACGT'), ('D', 'GCGT')],
        tree,
        sm,
        m,
        None,
        use_ambiguities,
    )
    assert tlk.log_likelihood() == pytest.approx(-21.7658748626709)


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
    )
    assert tlk.log_likelihood() == pytest.approx(-14.041006507122091)
    assert len(tlk.gradient()) == 4
    tree.set_parameters([0.01, 0.5, 0.2])
    assert tlk.log_likelihood() == pytest.approx(-15.25333438550469)
