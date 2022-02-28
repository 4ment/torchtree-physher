import pytest

import physherpy


def test_weibull_JC69():
    m = physherpy.physher.WeibullSiteModel(0.1, 4, None)
    sm = physherpy.physher.JC69()
    tree = physherpy.physher.TreeModel(
        "(A:0.1,(B:0.1,C:0.2):0.1);", ['A', 'B', 'C'], None
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
