import numpy as np
import pytest

import physherpy


def test_constant_coalescent():
    tree = physherpy.physher.ReparameterizedTimeTreeModel(
        "(((A,B),C),D);", ['A', 'B', 'C', 'D'], [0.0, 0.0, 0.0, 0.0]
    )
    tree.set_parameters(np.array([2.0 / 6.0, 6.0 / 12.0, 12.0]))
    constant = physherpy.physher.ConstantCoalescentModel(3.0, tree)
    assert constant.log_likelihood() == pytest.approx(-13.295836866)
    np.testing.assert_allclose(
        constant.gradient(), np.array([2.33333333, -1.0, -0.66666667, -0.33333333])
    )

    # physherpy.ReparameterizedTimeTreeModel(
    #     'id',
    # )
    # physherpy.ConstantCoalescentModel(3.0)


def test_skyride():
    tree = physherpy.physher.ReparameterizedTimeTreeModel(
        "(((A,B),C),D);", ['A', 'B', 'C', 'D'], [0.0, 0.0, 0.0, 0.0]
    )
    tree.set_parameters(np.array([2.0 / 6.0, 6.0 / 12.0, 12.0]))
    skyride = physherpy.physher.PiecewiseConstantCoalescentModel([3.0, 10.0, 4.0], tree)
    assert skyride.log_likelihood() == pytest.approx(-11.487491742782)
    np.testing.assert_allclose(
        skyride.gradient(),
        np.array([1.0, 0.02, 0.125, -1.7, -0.05, -0.25]),
    )


def test_skyride_heterochronous():
    tree = physherpy.physher.ReparameterizedTimeTreeModel(
        "((A,B),(C,D));", ['A', 'B', 'C', 'D'], [0.0, 1.0, 1.0, 0.0]
    )
    tree.set_parameters(np.array([2 / 3, 1 / 3, 4.0]))  # heights: 3.0, 10.0, 4.0
    skyride = physherpy.physher.PiecewiseConstantCoalescentModel([3.0, 10.0, 4.0], tree)
    assert skyride.log_likelihood() == pytest.approx(-7.67082507611538)
    np.testing.assert_allclose(
        skyride.gradient(),
        np.array([0.444444444, -0.07, -0.1875, -0.05, -1.7, -0.25]),
    )


def test_skygrid():
    tree = physherpy.physher.ReparameterizedTimeTreeModel(
        "(((A,B),C),D);", ['A', 'B', 'C', 'D'], [0.0, 0.0, 0.0, 0.0]
    )
    tree.set_parameters(np.array([2.0 / 6.0, 6.0 / 12.0, 12.0]))
    skyrgrid = physherpy.physher.PiecewiseConstantCoalescentGridModel(
        [3.0, 10.0, 4.0, 2.0, 3.0], tree, 10.0
    )
    assert skyrgrid.log_likelihood() == pytest.approx(-11.8751856)
    np.testing.assert_allclose(
        skyrgrid.gradient(),
        np.array(
            [1.16666667, 0.0750, 0.03125, 0.625, -0.11111111, -1.0, -0.5, -0.33333333]
        ),
    )


@pytest.mark.parametrize(
    "cutoff,expected,gradient",
    [
        (
            10.0,
            -19.594893640219844,
            [
                -0.23254415793482963,
                -0.03863268357286533,
                -0.0024726079643130304,
                0.0,
                -0.00012334888416770068,
                -0.36787944117144233,
                -0.09957413673572788,
                -0.0024787521766663585,
                -0.00012340980408667953,
            ],
        ),
        (
            18.0,
            -14.918634593243764,
            [
                -0.059082466159821156,
                -0.04606894010286441,
                9.216318529992316e-06,
                -0.0003351812899657137,
                0.0,
                -0.36787944117144233,
                -0.7357588823428847,
                -0.049787068367863944,
                -0.00033546262790251185,
            ],
        ),
    ],
)
def test_skygrid_heterochronous(cutoff, expected, gradient):
    tree = physherpy.physher.ReparameterizedTimeTreeModel(
        "((((A,B),C),D),E);", ['A', 'B', 'C', 'D', 'E'], [0.0, 1.0, 2.0, 3.0, 12.0]
    )
    lower = np.array([1.0, 2.0, 3.0, 12.0])
    ratios = convert(np.array([1.5, 4.0, 6.0, 16.0]), lower)
    tree.set_parameters(np.array(ratios))
    thetas_log = np.array([1.0, 3.0, 6.0, 8.0, 9.0])
    thetas = np.exp(thetas_log)
    skyrgrid = physherpy.physher.PiecewiseConstantCoalescentGridModel(
        thetas, tree, cutoff
    )
    assert skyrgrid.log_likelihood() == pytest.approx(expected)
    np.testing.assert_allclose(
        skyrgrid.gradient(),
        np.array(gradient),
    )


def convert(heights, lower):
    ratios = heights.copy()
    ratios[..., 2] = (heights[..., 2] - lower[..., 2]) / (
        heights[..., 3] - lower[..., 2]
    )
    ratios[..., 1] = (heights[..., 1] - lower[..., 1]) / (
        heights[..., 2] - lower[..., 1]
    )
    ratios[..., 0] = (heights[..., 0] - lower[..., 0]) / (
        heights[..., 1] - lower[..., 0]
    )
    return ratios


def inv_convert(ratios, lower):
    heights = ratios.copy()
    heights[..., 2] = lower[..., 2] + ratios[..., 2] * (heights[..., 3] - lower[..., 2])
    heights[..., 1] = lower[..., 1] + ratios[..., 1] * (heights[..., 2] - lower[..., 1])
    heights[..., 0] = lower[..., 0] + ratios[..., 0] * (heights[..., 1] - lower[..., 0])
    return heights
