import numpy as np
import pytest

import physherpy


def test_weibull_JC69():
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


def test_constant_coalescent():
    tree = physherpy.physher.ReparameterizedTimeTreeModel(
        "(((A,B),C),D);", ['A', 'B', 'C', 'D'], [0.0, 0.0, 0.0, 0.0]
    )
    tree.set_parameters(np.array([2.0 / 6.0, 6.0 / 12.0, 12.0]))
    constant = physherpy.physher.ConstantCoalescentModel(3.0, tree)
    print(constant.log_likelihood())
    assert constant.log_likelihood() == pytest.approx(-13.295836866)
    np.testing.assert_allclose(
        constant.gradient(), np.array([2.33333333, -1.0, -0.66666667, -0.33333333])
    )
