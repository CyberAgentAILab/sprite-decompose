from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np


class AlphaL1:
    def __init__(self, optimize_order: bool = True, order_axis: int = 0) -> None:
        assert order_axis == 0, "Currently only support order_axis=0"
        self.optimize_order = optimize_order
        self.order_axis = order_axis

    def __call__(
        self, inputs: np.ndarray, targets: np.ndarray, order: Optional[List[int]] = None
    ) -> Tuple[float, Optional[List[int]]]:
        """Calculate L1 loss for alpha channel with optional layer order optimization.
        inputs: Array of shape (..., 4)
        targets: Array of shape (..., 4)
        order: If not None, calculate L1 loss with the specified order.
        >>>
        loss value, optimized order
        """
        assert inputs.shape == targets.shape
        assert inputs.shape[-1] == 4
        if self.optimize_order:
            if order is None:
                order_list = list(permutations(range(inputs.shape[self.order_axis])))
                score_list = [alpha_l1(inputs[list(order)], targets) for order in order_list]
                best_idx = np.argmin(score_list)
                best_order = order_list[best_idx]
                return score_list[best_idx], best_order
            else:
                order = order or list(range(inputs.shape[self.order_axis]))
                return alpha_l1(inputs[list(order)], targets), order
        else:
            return alpha_l1(inputs, targets), None


class RGBL1AlphaWeight:
    def __init__(self, optimize_order: bool = True, order_axis: int = 0, average_type: str = "batch") -> None:
        assert order_axis == 0, "Currently only support order_axis=0"
        assert average_type in ["batch", "point"]
        self.optimize_order = optimize_order
        self.order_axis = order_axis
        self.average_type = average_type

    def __call__(
        self, inputs: np.ndarray, targets: np.ndarray, order: Optional[List[int]] = None
    ) -> Tuple[float, Optional[List[int]]]:
        """Calculate L1 loss for RGB channels weighted by alpha with optional layer order optimization.
        inputs: Array of shape (..., 4)
        targets: Array of shape (..., 4)
        order: If not None, calculate L1 loss with the specified order.
        >>>
        loss value, optimized order
        """
        assert inputs.shape == targets.shape
        assert inputs.shape[-1] == 4
        if self.optimize_order:
            if order is None:
                order_list = list(permutations(range(inputs.shape[self.order_axis])))
                score_list = [
                    rgb_l1_alpha_weight(inputs[list(order)], targets, average_type=self.average_type)
                    for order in order_list
                ]
                best_idx = np.argmin(score_list)
                best_order = order_list[best_idx]
                return score_list[best_idx], best_order
            else:
                order = order or list(range(inputs.shape[self.order_axis]))
                return rgb_l1_alpha_weight(inputs[list(order)], targets, average_type=self.average_type), order
        else:
            return rgb_l1_alpha_weight(inputs, targets, average_type=self.average_type), None


def rgb_l1_alpha_weight(y: np.ndarray, t: np.ndarray, average_type: str = "batch") -> float:
    assert average_type in ["batch", "point"]
    if average_type == "point":
        loss_rgb = np.abs(y[..., :-1] - t[..., :-1]).mean(axis=-1)
        loss_rgb = (loss_rgb * t[..., -1]).sum() / t[..., -1].sum()
    elif average_type == "batch":
        n = y.shape[0]
        loss_rgb = np.abs(y[..., :-1] - t[..., :-1]).mean(axis=-1)
        loss_rgb = loss_rgb.reshape(n, -1)
        alpha = t[..., -1].reshape(n, -1)
        loss_rgb = ((loss_rgb * alpha).sum(axis=-1) / alpha.sum(axis=-1)).mean()
    return loss_rgb


def alpha_l1(y: np.ndarray, t: np.ndarray) -> float:
    return np.abs(y[..., -1] - t[..., -1]).mean()
