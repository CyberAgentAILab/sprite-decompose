from typing import Union

import numpy as np
import torch as tr


def mat_scale_tr(scale: Union[tr.Tensor, np.ndarray]) -> tr.Tensor:
    """Genaerates a 3x3 matrix for scaling.
    scale: Tensor with shape (..., 2)
    >>>
    Tensor with shape (..., 3, 3)
    """
    if not isinstance(scale, tr.Tensor):
        scale = tr.tensor(scale)
    origin_shape = scale.shape
    scale_flat = scale.reshape(-1, 2)
    n = len(scale_flat)
    mat = tr.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).repeat(n, 1, 1).float().to(scale.device)
    mat[:, 0, 0] = scale_flat[:, 0]
    mat[:, 1, 1] = scale_flat[:, 1]
    return mat.reshape(list(origin_shape[:-1]) + [3, 3])


def mat_shift_tr(shift: Union[tr.Tensor, np.ndarray]) -> tr.Tensor:
    """Generates a 3x3 matrix for shifting.
    shift: Tensor with shape (..., 2)
    >>>
    Tensor with shape (..., 3, 3)
    """
    if not isinstance(shift, tr.Tensor):
        shift = tr.tensor(shift)
    origin_shape = shift.shape
    shift_flat = shift.reshape(-1, 2)
    n = len(shift_flat)
    mat = tr.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).repeat(n, 1, 1).float().to(shift.device)
    mat[:, 0, 2] = shift_flat[:, 0]
    mat[:, 1, 2] = shift_flat[:, 1]
    return mat.reshape(list(origin_shape[:-1]) + [3, 3])


def mat_rot_tr(rad: Union[tr.Tensor, np.ndarray]) -> tr.Tensor:
    """Generates a 3x3 matrix for rotation.
    rad: Tensor with shape (...)
    >>>
    Tensor with shape (..., 3, 3)
    """
    if not isinstance(rad, tr.Tensor):
        rad = tr.tensor(rad)
    origin_shape = rad.shape
    rad_flat = rad.reshape(-1)
    n = len(rad_flat)
    mat = tr.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).repeat(n, 1, 1).float().to(rad.device)
    sin, cos = tr.sin(rad_flat), tr.cos(rad_flat)
    mat[:, 0, 0] = cos
    mat[:, 0, 1] = -sin
    mat[:, 1, 0] = sin
    mat[:, 1, 1] = cos
    return mat.reshape(list(origin_shape) + [3, 3])
