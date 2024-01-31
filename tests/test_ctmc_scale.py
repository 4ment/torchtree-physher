import numpy as np
import pytest

from torchtree_physher.physher import CTMCScaleModel, ReparameterizedTimeTreeModel
from torchtree_physher.physher.tree_transform_flags import RATIO


def test_ctmc_scale():
    tree_model = ReparameterizedTimeTreeModel(
        "((((A_0:1.5,B_1:0.5):2.5,C_2:2):2,D_3:3):10,E_12:4);",
        ["A_0", "B_1", "C_2", "D_3", "E_12"],
        [0.0, 1.0, 2.0, 3.0, 12.0],
        RATIO,
    )
    ratios = convert(np.array([1.5, 4.0, 6.0, 16.0]), np.array([1.0, 2.0, 3.0, 12.0]))
    tree_model.set_parameters(ratios)

    ctmc_scale = CTMCScaleModel([0.001], tree_model)
    assert 4.475351922659342 == pytest.approx(ctmc_scale.log_likelihood(), 0.00001)

    np.testing.assert_allclose(
        np.array(
            [
                0.05582353100180626,
                0.08683660626411438,
                0.38301146030426025,
                0.04401470720767975,
                -525.5,
            ]
        ),
        ctmc_scale.gradient(),
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
