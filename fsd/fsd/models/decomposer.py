import os
import os.path as osp
import time
from itertools import permutations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import fsspec
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from einops import rearrange, repeat
from fsd.loss_func.build_loss_func import build_loss_funcs_from_cfg
from fsd.models.matrix_utils import mat_scale_tr, mat_shift_tr
from fsd.models.segmentation import segmentation_tam_key_box_prompt
from fsd.models.sprites import Sprites
from fsd.utils.io import save_json
from fsd.utils.log import get_logger
from omegaconf import DictConfig
from tqdm import tqdm

from .tex_gen import TexUNet

logger = get_logger(name=__name__)


class SpritesDecomposer(Sprites):
    """Class for decomposing sprites into layers. Subclass of Sprites."""

    def __init__(
        self,
        cfg: DictConfig,
        target: Dict[str, tr.Tensor],
        prompt: Dict[str, Any],
        out_dir: str,
        resume: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        cfg: config file
        target:
          - "frames": the target video to decompose (n_frames, height, width, 4).
          - "layers": the ground truth layers (n_layers, n_frames, height, width, 4).
                      If calculating evaluation metrics, it is required.
        prompt:
          - "key_box": array of bounding boxes for each layer (n_layers, 4).
          - "key_index": array of key indices for each layer (n_layers).
        out_dir: output directory
        resume: If True, resume training from the last checkpoint.
        device: "cpu" or "cuda".
        """
        start = time.time()
        self.target = target
        self.target_np = {k: v.detach().cpu().numpy() for k, v in target.items()}
        n_frames, height, width, c = target["frames"].shape
        n_foregrounds = len(prompt["key_box"])
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device
        self.iter = 0
        self.optimization_time = 0
        self.eval_log = []
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        matrices = repeat(tr.eye(3), "i j -> f t (i j)", f=n_foregrounds + 1, t=n_frames)
        # opacity = tr.logit(tr.ones([n_foregrounds + 1, n_frames]) - 1e-6)
        opacity = tr.ones([n_foregrounds + 1, n_frames])
        layer_order = list(range(n_foregrounds + 1))
        textures = tr.rand([n_foregrounds + 1, 4, cfg.texture_size[0], cfg.texture_size[1]])
        super().__init__(
            textures, matrices, opacity, height, width, layer_order, trainable=True, opacity_sigmoid=True, texture_sigmoid=True
        )

        # Overwrite textures
        if cfg.use_texture_model:
            del self.textures
            self.texture_net = TexUNet(n_foregrounds + 1, target_shape=cfg.texture_size, **cfg.texture_net)
        self.to(device)

        self.loss_fn = build_loss_funcs_from_cfg(cfg.loss_fn)
        self.optimizer = getattr(optim, cfg.optimizer.name)(self.parameters(), **cfg.optimizer.params)

        if resume and osp.exists(osp.join(out_dir, "last_checkpoint.pth")):
            self.load_checkpoint(strict_config=True)
        elif cfg.init_textures or cfg.init_matrices:  # Initialization
            self._initialize(
                target["frames"].cpu().numpy(),
                prompt,
                init_textures=cfg.init_textures,
                init_matrices=cfg.init_matrices,
            )
            self.optimization_time += time.time() - start
        self._infer_textures()

    def _initialize(
        self, frames: np.ndarray, prompt: Dict[str, Any], init_textures: bool = True, init_matrices: bool = True
    ) -> None:
        logger.info("segmentation by TAM...")
        masks = []
        for i in range(self.num_layers - 1):
            key_box = prompt["key_box"][i]
            key_index = prompt["key_index"][i]
            masks_layer = segmentation_tam_key_box_prompt(
                frames, key_box, key_index=key_index, require_unnormalize=True, device=self.device
            )
            masks.append(masks_layer)
        masks = np.array(masks)
        bboxes = generate_boxes_from_masks(masks)
        if init_matrices:
            bboxes_ = np.array([fill_bboxes(b) for b in bboxes]) * 2 - 1
            ymin, ymax, xmin, xmax = (bboxes_[..., 0], bboxes_[..., 1], bboxes_[..., 2], bboxes_[..., 3])
            scale = np.stack([(xmax - xmin) / 2, (ymax - ymin) / 2], axis=2)
            shift = np.stack([(xmin + xmax) / 2, (ymin + ymax) / 2], axis=2)
            fg_matrices = mat_shift_tr(shift) @ mat_scale_tr(scale)
            bg_matrix = tr.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, self.num_frames, 1, 1)
            matrices = rearrange(tr.cat([bg_matrix, fg_matrices], dim=0), "f t i j -> f t (i j)")
            self.matrices.data = matrices.to(self.device)
        if init_textures:
            # Generate rough bg texture: inpaint unvisible area with the mean value
            fg_mask = masks.any(0)
            not_fg_mask = np.logical_not(fg_mask)[..., None]
            with np.errstate(invalid="ignore"):
                bg = (frames * not_fg_mask).sum(axis=0) / not_fg_mask.sum(axis=0)
            if np.isnan(bg).all():  # If all pixels are nan, generate random texture
                bg = np.random.rand(4, self.cfg.texture_size[0], self.cfg.texture_size[1])
            else:
                bg = np.array([np.nan_to_num(bg[..., i], nan=np.nanmean(bg[..., i])) for i in range(4)])
                bg = cv2.resize(bg.transpose(1, 2, 0), (self.cfg.texture_size[0], self.cfg.texture_size[1]))
                bg = bg.transpose(2, 0, 1)

            # Generate rough fg textures: crop masked area and take mean
            fgs = []
            h, w = self.height, self.width
            for i in range(self.num_layers - 1):
                croppeds = []
                for j in range(self.num_frames):
                    if np.isnan(bboxes[i, j]).any():
                        continue
                    ymin, ymax, xmin, xmax = bboxes[i, j]
                    ymin, ymax, xmin, xmax = int(ymin * h), int(ymax * h), int(xmin * w), int(xmax * w)
                    cropped = np.copy(frames[j, ymin:ymax, xmin:xmax])
                    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                        continue
                    cropped_mask = masks[i, j, ymin:ymax, xmin:xmax]
                    cropped[..., -1] = cropped_mask
                    croppeds.append(cv2.resize(cropped, (self.cfg.texture_size[0], self.cfg.texture_size[1])))
                if len(croppeds) > 0:
                    fgs.append(np.array(croppeds).mean(0).transpose(2, 0, 1))
                else:
                    logger.info(f"fg {i} is not visible, generate random texture")
                    fgs.append(np.random.rand(4, self.cfg.texture_size[0], self.cfg.texture_size[1]))

            textures = tr.tensor(np.array([bg] + fgs)).float()
            if self.texture_net is not None:
                self.texture_net.codes = nn.Parameter(textures.unsqueeze(0).to(self.device))
            else:
                textures = rearrange(textures, "l c h w -> l h w c")
                textures[textures == 0] = 1e-6
                textures[textures == 1] = 1 - 1e-6
                self.textures = nn.Parameter(tr.logit(textures).to(self.device))

    def _infer_textures(self) -> None:
        self.textures = rearrange(self.texture_net()["texs"][0], "l c h w -> l h w c")

    def _compute_loss(self, optimize_layer_order: bool = False) -> Tuple[Dict[str, tr.Tensor], Dict[str, tr.Tensor]]:
        layers = self.render_layers(inference=True)

        if optimize_layer_order:  # NOTE: bg is always layers[0]
            layer_order_list = [[0] + list(o) for o in permutations(range(1, self.num_layers))]
        else:
            layer_order_list = [self.layer_order]
        best_loss_sum = tr.inf
        for layer_order in layer_order_list:
            recon = self.render(layers=layers, layer_order=layer_order, inference=False)
            loss = self.loss_fn({"frames": recon}, self.target, labels=["main"])
            loss_sum = sum(loss.values())
            if loss_sum < best_loss_sum:
                best_loss_sum = loss_sum
                best_metric = loss
                best_recon = recon
                best_layer_order = layer_order
        self.layer_order = best_layer_order
        best_layers = layers[best_layer_order]
        output = {"frames": best_recon, "layers": best_layers}
        return output, best_metric

    def render_layers(self, inference: bool = True) -> tr.Tensor:
        if inference and self.texture_net is not None:
            self._infer_textures()
        return super().render_layers()

    def render(
        self, layers: Optional[tr.Tensor] = None, layer_order: Optional[List[int]] = None, inference: bool = True
    ) -> tr.Tensor:
        if inference and self.texture_net is not None:
            self._infer_textures()
        return super().render(layers=layers, layer_order=layer_order)

    def fit(
        self,
        n_iters: int = 1000,
        max_time: Optional[float] = None,
        eval_interval: int = 100,
        save_interval: int = 100,
        metric_fn: Optional[Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray]], Dict[str, float]]] = None,
        verbose: bool = True,
    ) -> None:
        logger.info(f"Start optimization (n_iters: {n_iters}, max_time: {max_time} sec)")
        max_time = max_time or float("inf")
        with tqdm(range(self.iter + 1, n_iters + 1), ncols=0, leave=True, total=n_iters, initial=self.iter) as pbar:
            for i in pbar:
                self.iter = i
                start = time.time()
                if (i < self.cfg.n_iters_layer_order_optimize) or (i % self.cfg.layer_order_optimize_interval == 0):
                    optimize_layer_order = True
                else:
                    optimize_layer_order = False
                output, loss = self._compute_loss(optimize_layer_order=optimize_layer_order)
                total_loss = sum(loss.values())
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.optimization_time += time.time() - start

                metric = {"iteration": i, "optimization_time": self.optimization_time}

                if (i % eval_interval == 0 or (i) == n_iters or (max_time < self.optimization_time)) and metric_fn is not None:
                    loss_dict = {k: v.detach().cpu().numpy() for k, v in loss.items()}
                    loss_dict["loss"] = sum(loss_dict.values())
                    metric.update(loss_dict)
                    output_np = {k: v.detach().cpu().numpy() for k, v in output.items()}
                    metric.update(metric_fn(output_np, self.target_np))
                self.eval_log.append(metric)
                if i % save_interval == 0 or (i) == n_iters or (max_time < self.optimization_time):
                    self.save_checkpoint()
                    self.save(osp.join(self.out_dir, "sprites", f"iter_{i:09}"), overwrite=True)
                    self.save_visualization(osp.join(self.out_dir, "sprites", f"iter_{i:09}.png"))
                    save_json(self.eval_log, osp.join(self.out_dir, "progress.json"))
                if verbose:
                    pbar.set_postfix(metric)
                if self.optimization_time > max_time:
                    logger.info(f"max_time reached: {max_time} sec, now: {self.optimization_time} sec")
                    break

    def export_to_checkpoint(self) -> Dict[str, Any]:
        return {
            "cfg": self.cfg,
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "layer_order": self.layer_order,
            "out_dir": self.out_dir,
            "iteration": self.iter,
            "eval_log": self.eval_log,
            "optimization_time": self.optimization_time,
        }

    def save_checkpoint(self, name: str = "last_checkpoint.pth") -> None:
        checkpoint = self.export_to_checkpoint()
        with fsspec.open(osp.join(self.out_dir, name), "bw") as f:
            tr.save(checkpoint, f)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None, strict_config: bool = True) -> None:
        checkpoint_path = checkpoint_path or osp.join(self.out_dir, "last_checkpoint.pth")
        with fsspec.open(checkpoint_path, "br") as f:
            checkpoint = tr.load(f, map_location=self.device)
        if self.cfg != checkpoint["cfg"]:
            if strict_config:
                assert self.cfg == checkpoint["cfg"], "Config is different, but strict_config is True"
            else:
                logger.warning("Config is different, but strict_config is False")

        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.layer_order = checkpoint["layer_order"]
        self.out_dir = checkpoint["out_dir"]
        self.iter = checkpoint["iteration"]
        self.optimization_time = checkpoint["optimization_time"]
        self.eval_log = checkpoint["eval_log"]
        logger.info(f"Checkpoint loaded from {checkpoint_path} (iter {self.iter})")


def mask2bbox(mask: np.ndarray, normalize: bool = True, th: float = 0) -> List[float]:
    ys, xs = np.where(mask > th)
    if len(ys) == 0:
        return [np.nan] * 4
    h, w = mask.shape
    box = [min(ys), max(ys) + 1, min(xs), max(xs) + 1]  # for zero size box
    if normalize:
        return [box[0] / h, box[1] / h, box[2] / w, box[3] / w]
    else:
        return box


def generate_boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """
    masks: (..., H, W)
    """
    h, w = masks.shape[-2:]
    bboxes = np.array([mask2bbox(mask) for mask in masks.reshape(-1, h, w)])
    return bboxes.reshape(masks.shape[:-2] + (4,))


def fill_bboxes(bboxes: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    if np.isnan(bboxes).all():
        return np.array([[0, 1, 0, 1]] * len(bboxes))
    bboxes = bboxes if isinstance(bboxes, np.ndarray) else np.array(bboxes)
    return np.array([interp(corners) for corners in bboxes.transpose(1, 0)]).transpose(1, 0)


def interp(x: np.ndarray) -> np.ndarray:
    new_index = np.arange(len(x))
    filled_indices = ~np.isnan(x)
    return np.interp(new_index, new_index[filled_indices], x[filled_indices])
