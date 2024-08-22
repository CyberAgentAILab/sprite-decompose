import os
import os.path as osp
from typing import List, Optional, Tuple, Union

import numpy as np
import torch as tr
import torch.nn as nn
from einops import rearrange, repeat
from fsd.utils.io import read_image_pil, read_json, save_frames_as_apng, save_image_pil, save_json
from fsd.utils.vis import insert_checker_board_bg
from PIL import Image
from torch import Tensor


def warp_affine_torch(
    x: tr.Tensor, mats: tr.Tensor, height: int, width: int, inv: bool = True, align_corners: bool = False
) -> tr.Tensor:
    """
    x: (N, C, H, W)
    mats: (N, 3, 3)
    inv: Defaults to True.
    >>>
    tensor: (N, C, height, width)
    """
    n, c, h, w = x.shape
    if inv:
        mats = tr.linalg.inv(mats)
    grid = tr.nn.functional.affine_grid(mats[:, :2], (n, c, height, width), align_corners=align_corners)
    warped = tr.nn.functional.grid_sample(x, grid, align_corners=align_corners)
    return warped


def alpha_blend_torch_core(fg_image: Tensor, bg_image: Tensor, norm_value: float = 1) -> Tensor:
    """
    fg_image: (..., H, W, C)
    bg_image: (..., H, W, C)
    >>>
    tensor: (..., H, W, C)
    """
    assert fg_image.shape == bg_image.shape, f"fg_image.shape: {fg_image.shape}, bg_image.shape: {bg_image.shape}"
    fg_image = fg_image.float()
    bg_image = bg_image.float()

    fg_rgb = fg_image[..., :3]
    bg_rgb = bg_image[..., :3]
    fg_alpha = fg_image[..., 3:4] / norm_value
    bg_alpha = bg_image[..., 3:4] / norm_value
    if fg_alpha.max() > 1 or fg_alpha.min() < 0 or bg_alpha.max() > 1 or bg_alpha.min() < 0:
        raise ValueError("Alpha values should be normalized to [0, 1] by norm_value.")
    alpha = fg_alpha + bg_alpha * (1 - fg_alpha)  # source over
    mask = (alpha > 0)[..., 0]
    rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
    alpha_masked = alpha[tr.where(mask)]
    rgb_masked = rgb[tr.where(mask)]
    rgba_masked = tr.cat([rgb_masked, alpha_masked * norm_value], dim=-1)
    output = tr.zeros_like(fg_image)
    output[tr.where(mask)] = rgba_masked
    return output


def render_layers_torch_core(textures: Tensor, matrices: Tensor, opacity: Tensor, height: int, width: int) -> Tensor:
    """
    textures: (n_layers, H, W, C)
    matrices: (n_layers, T, 9)
    opacity: (n_layers, T)
    normalize_opacity: If True, opacity will be normalized to [0, 1].
    >>>
    tensor: (n_layers, T, H, W, C)
    """
    assert (textures.max() <= 1.0) and (textures.min() >= 0), "Textures should be normalized to [0, 1]."
    assert (opacity.max() <= 1.0) and (opacity.min() >= 0), "opacity should be normalized to [0, 1]."
    l, t, _ = matrices.shape  # noqa: E741
    tex_repeat = repeat(textures, "l h w c -> (l t) c h w", t=t)  # (n_layers * t, c, h, w)
    tex_warped = warp_affine_torch(tex_repeat, matrices.reshape(-1, 3, 3), height=height, width=width)
    tex_warped = rearrange(tex_warped, "(l t) c h w -> l t h w c", l=l)

    tex_warped_alpha = tex_warped[..., -1].clone()
    tex_warped[..., -1] = tex_warped_alpha * opacity[:, :, None, None]
    return tex_warped


class Sprites(nn.Module):
    """Class for rendering sprites by differentiable compositing with PyTorch functions."""

    def __init__(
        self,
        textures: Tensor,
        matrices: Tensor,
        opacity: Tensor,
        height: int,
        width: int,
        layer_order: Optional[List[int]] = None,
        trainable: bool = False,
        opacity_sigmoid: bool = False,
        texture_sigmoid: bool = False,
    ):
        """
        textures: (n_layers, H, W, C)
        matrices: (n_layers, T, 9)
        opacity: (n_layers, T)
        height: canvas height
        width: canvas width
        layer_order: order of layers to render. If None, render by the order of textures.
        trainable: If True, textures, matrices, and opacity will be trainable parameters.
        normalize_opacity: If True, opacity will be normalized to [0, 1].
        normalize_texture: If True, textures will be normalized to [0, 1].
        """
        super().__init__()
        textures = textures.float() if isinstance(textures, tr.Tensor) else tr.tensor(textures).float()
        matrices = matrices.float() if isinstance(matrices, tr.Tensor) else tr.tensor(matrices).float()
        opacity = opacity.float() if isinstance(opacity, tr.Tensor) else tr.tensor(opacity).float()
        if not texture_sigmoid and (textures.max() > 1 or textures.min() < 0):
            raise ValueError("Textures should be normalized to [0, 1] or texture_sigmoid should be True.")
        if not opacity_sigmoid and (opacity.max() > 1 or opacity.min() < 0):
            raise ValueError("Opacity should be normalized to [0, 1] or opacity_sigmoid should be True.")
        self.textures = textures if not trainable else nn.Parameter(textures.clone())
        self.matrices = matrices if not trainable else nn.Parameter(matrices.clone())
        self.opacity = opacity if not trainable else nn.Parameter(opacity.clone())
        self.layer_order = list(layer_order) if layer_order is not None else list(range(len(textures)))
        self.height = height
        self.width = width

        self.opacity_sigmoid = opacity_sigmoid
        self.texture_sigmoid = texture_sigmoid
        self.bg_transparent = False
        self.static_bg = True

    @property
    def num_frames(self) -> int:
        return self.matrices.shape[1]

    @property
    def num_layers(self) -> int:
        return self.matrices.shape[0]

    @property
    def size(self) -> Tuple[int, int]:
        return self.height, self.width

    def render_layers(self) -> Tensor:
        """
        >>>
        tensor: (n_layers, T, H, W, C)
        """
        layers = render_layers_torch_core(
            tr.sigmoid(self.textures) if self.texture_sigmoid else self.textures,
            self.matrices,
            tr.sigmoid(self.opacity) if self.opacity_sigmoid else self.opacity,
            height=self.height,
            width=self.width,
        )

        if not self.bg_transparent:
            layers[self.layer_order[0], :, :, :, -1] = 1
        return layers

    def render(self, layers: Optional[Tensor] = None, layer_order: Optional[List[int]] = None) -> Tensor:
        """
        >>>
        tensor: (T, H, W, C)
        """
        layers = self.render_layers() if layers is None else layers
        layer_order = self.layer_order if layer_order is None else layer_order
        layers = layers[layer_order]
        backdrop = layers[0]  # .clone()
        for layer in layers[1:]:
            backdrop = alpha_blend_torch_core(layer, backdrop)
        return backdrop

    def resize(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        ratio: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        if size is not None:
            size = size if isinstance(size, (list, tuple)) else [size, size]
            self.height, self.width = size
        else:
            assert ratio is not None, "if size is None, ratio should be specified."
            ratio = ratio if isinstance(ratio, (list, tuple)) else [ratio, ratio]
            self.height = round(self.height * ratio[0])
            self.width = round(self.width * ratio[1])

    def save(self, path: str, overwrite: bool = False):
        if osp.exists(osp.join(path, "textures")):
            assert overwrite, f"{path} already exists, please set overwrite=True."
        else:
            os.makedirs(osp.join(path, "textures"))
        texture_paths = [osp.join("textures", f"{i}.png") for i in range(len(self.textures))]

        state = {
            "matrices": self.matrices,
            "opacity": self.opacity if not self.opacity_sigmoid else tr.sigmoid(self.opacity),
            "height": self.height,
            "width": self.width,
            "layer_order": self.layer_order,
            "texture_paths": texture_paths,
        }
        save_json(state, osp.join(path, "params.json"))
        for i in range(self.num_layers):
            texture = self.textures[i] if not self.texture_sigmoid else tr.sigmoid(self.textures[i])
            texture_np = texture.detach().cpu().numpy()
            if i == self.layer_order[0] and not self.bg_transparent:
                texture_np[..., -1] = 1
            save_image_pil(Image.fromarray((texture_np * 255).astype(np.uint8)), osp.join(path, texture_paths[i]))

    def save_visualization(self, path: str, pad_size: int = 5):
        layers = self.render_layers().cpu().detach()
        frames = self.render(layers=layers).cpu().detach()
        layers = (layers * 255).numpy().astype(np.uint8)
        frames = (frames * 255).numpy().astype(np.uint8)
        layers = np.array([[insert_checker_board_bg(f) for f in layer] for layer in layers])
        frames = np.pad(frames, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        layers = np.pad(layers, ((0, 0), (0, 0), (0, 0), (0, pad_size), (0, 0)))
        vis = rearrange(np.concatenate([frames[None], layers], axis=0), "l t h w c -> t h (l w) c")
        save_frames_as_apng(vis, path)

    @classmethod
    def open(cls, path: str) -> "Sprites":
        state = read_json(osp.join(path, "params.json"))
        textures = [np.array(read_image_pil(osp.join(path, p))).astype(np.float32) for p in state["texture_paths"]]
        return cls(
            tr.tensor(np.array(textures)) / 255,
            matrices=state["matrices"],
            opacity=state["opacity"],
            height=state["height"],
            width=state["width"],
            layer_order=state["layer_order"],
        )
