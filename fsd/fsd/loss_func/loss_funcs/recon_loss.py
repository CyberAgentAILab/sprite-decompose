import torch as tr
from torch import Tensor


class AlphaL1Loss:
    def __init__(self, weight_alpha: float = 1) -> None:
        self.weight_alpha = weight_alpha

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return alpha_l1_loss(inputs, targets, weight_alpha=self.weight_alpha)


def alpha_l1_loss(y: Tensor, t: Tensor, weight_alpha: float = 1) -> Tensor:
    """
    y: Tensor of shape (..., 4)
    t: Tensor of shape (..., 4)
    """
    base_loss = tr.nn.functional.l1_loss
    loss_alpha = base_loss(y[..., -1], t[..., -1])
    loss_rgb = tr.sum(base_loss(y[..., :-1], t[..., :-1], reduction="none").mean(dim=-1) * t[..., -1]) / tr.sum(
        t[..., -1:]
    )
    return loss_alpha * weight_alpha + loss_rgb


class AlphaL2Loss:
    def __init__(self, weight_alpha: float = 1) -> None:
        self.weight_alpha = weight_alpha

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return alpha_l2_loss(inputs, targets, weight_alpha=self.weight_alpha)


def alpha_l2_loss(y: Tensor, t: Tensor, weight_alpha: float = 1) -> Tensor:
    """
    y: Tensor of shape (..., 4)
    t: Tensor of shape (..., 4)
    """
    base_loss = tr.nn.functional.mse_loss
    loss_alpha = base_loss(y[..., -1], t[..., -1])
    loss_rgb = tr.sum(base_loss(y[..., :-1], t[..., :-1], reduction="none").mean(dim=-1) * t[..., -1]) / tr.sum(
        t[..., -1:]
    )
    return loss_alpha * weight_alpha + loss_rgb
