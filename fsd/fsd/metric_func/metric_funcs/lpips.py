from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np
import torch as tr

import lpips


class LPIPSAlphaWeight:
    def __init__(
        self, net: str = "alex", optimize_order: bool = True, order_axis: int = 0, average_type: str = "batch"
    ) -> None:
        assert order_axis == 0, "Currently only support order_axis=0"
        assert average_type in ["batch", "point"]
        self.optimize_order = optimize_order
        self.order_axis = order_axis
        self.net = lpips.LPIPS(net=net, spatial=True, verbose=True)
        self.average_type = average_type

    def score(self, y: np.ndarray, t: np.ndarray, value_range: Tuple[int, int] = (0, 1)) -> float:
        assert y.shape == t.shape
        assert y.shape[-1] == 4
        n = y.shape[0]
        h, w, _ = y.shape[-3:]
        y = y.reshape(-1, h, w, 4)
        t = t.reshape(-1, h, w, 4)
        y_rgb_weighted = y[..., :-1] * t[..., -1:]  # use gt for mask
        t_rgb_weighted = t[..., :-1] * t[..., -1:]
        y_rgb_weighted_normed = y_rgb_weighted * 2 / (value_range[1] - value_range[0]) - 1
        t_rgb_weighted_normed = t_rgb_weighted * 2 / (value_range[1] - value_range[0]) - 1
        y_tensor = tr.from_numpy(y_rgb_weighted_normed).permute(0, 3, 1, 2).float()
        t_tensor = tr.from_numpy(t_rgb_weighted_normed).permute(0, 3, 1, 2).float()
        with tr.no_grad():
            score_map = self.net.forward(y_tensor, t_tensor).numpy()[:, 0]
        if self.average_type == "point":
            return np.sum(score_map * t[..., -1]) / np.sum(t[..., -1])
        elif self.average_type == "batch":
            score_map = score_map.reshape(n, -1)
            alpha = t[..., -1].reshape(n, -1)
            score_ = (score_map * alpha).sum(axis=-1) / alpha.sum(axis=-1)
            return score_.mean()

    def __call__(
        self, inputs: np.ndarray, targets: np.ndarray, order: Optional[List[int]] = None
    ) -> Tuple[float, Optional[List[int]]]:
        """Calculate LPIPS loss for RGB channels weighted by alpha with optional layer order optimization.
        inputs: Array of shape (..., H, W, 4)
        targets: Array of shape (..., H, W, 4)
        order: If not None, calculate LPIPS loss with the specified order.
        >>>
        loss value, optimized order
        """

        if self.optimize_order:
            if order is None:
                order_list = list(permutations(range(inputs.shape[self.order_axis])))
                score_list = [self.score(inputs[list(order)], targets, value_range=(0, 1)) for order in order_list]
                best_idx = np.argmin(score_list)
                best_order = order_list[best_idx]
                return score_list[best_idx], best_order
            else:
                order = order or list(range(inputs.shape[self.order_axis]))
                return self.score(inputs[list(order)], targets, value_range=(0, 1)), order
        else:
            return self.score(inputs, targets, value_range=(0, 1)), None
