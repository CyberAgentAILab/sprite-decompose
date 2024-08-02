import os.path as osp
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import torch as tr
from fsd.metric_func.build_metric_funcs import build_metric_funcs_from_cfg
from fsd.models.decomposer import SpritesDecomposer, mask2bbox
from fsd.models.sprites import Sprites
from fsd.utils.io import save_json
from fsd.utils.log import get_logger

logger = get_logger(name=__name__)


def simulate_key_boxes(layers: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Similate key boxes from the layers.
    layers (array): Array of shape (L, T, H, W, C) representing L layers, T frames, H height, W width, and C channels.
    k (int): extract the frame with k-th largest visible area.
    >>>
    key_boxes (array): Array of shape (L, 4) representing the key frame of each layer.
    key_indices (array): Array of shape (L,) representing the index of the key frame of each layer.
    """
    assert k > 0, "k must be greater than 0."
    fg_masks = layers[1:, ..., -1] > 0
    key_boxes = []
    key_indices = []
    n_layers = len(fg_masks)
    for i in range(n_layers):
        assert fg_masks[i].any(), f"Layer {i} has no visible area."
        mask_others = fg_masks[np.arange(n_layers) != i].any(axis=0)
        interection = fg_masks[i] & mask_others
        visible = fg_masks[i] & ~interection
        visible_area = visible.sum(axis=(1, 2))
        visible_area_max = np.sort(visible_area)[-k]
        if visible_area_max == 0:
            # if there is no visible area, choose the frame with the largest visible area
            visible_area = fg_masks[i].sum(axis=(1, 2))
            visible_area_max = visible_area.max()
        key_index = np.where(visible_area == visible_area_max)[0]
        # if there are multiple frames with the same visible area, choose the middle one
        key_index = key_index[len(key_index) // 2]
        key_boxes.append(mask2bbox(fg_masks[i, key_index]))
        key_indices.append(key_index)

    return np.array(key_boxes), np.array(key_indices)


def make_target_and_prompt(example: Dict[str, Any]) -> Tuple[Dict[str, tr.Tensor], Dict[str, tr.Tensor]]:
    textures_np = np.array([np.array(t) for t in example["texture"]])
    sprites = Sprites(
        textures_np / 255,
        matrices=example["matrix"],
        opacity=example["opacity"],
        height=example["canvas_height"],
        width=example["canvas_width"],
    )

    # Simulate box prompts
    layers = sprites.render_layers()
    key_box, key_index = simulate_key_boxes(layers, k=1)
    data = {
        "frames": sprites.render(),
        "layers": layers,
        "matrices": sprites.matrices,
        "opacity": sprites.opacity,
    }
    prompt = {"key_box": tr.tensor(key_box), "key_index": tr.tensor(key_index)}
    return data, prompt


def optimization(cfg: Any, timelimit_minute: int = None) -> None:
    # Dataset
    dataset = datasets.load_dataset(cfg.data.name, split=cfg.data.split)

    # Metric
    metric_fn = build_metric_funcs_from_cfg(cfg.metric_func)

    # Optimization
    timelimit_minute = cfg.engine.timelimit_minute if timelimit_minute is None else timelimit_minute
    for i, example in enumerate(dataset):
        target, prompt = make_target_and_prompt(example)
        template_id = example["id"]
        l, t, h, w, c = target["layers"].shape
        logger.info(f"{i + 1} / {len(dataset)}: {template_id}")
        logger.info(f"Video size: {t} frames, {h}x{w} pixels, {l} layers")

        out_dir = osp.join(cfg.out_dir, f"{template_id}")
        target = {k: v.to(cfg.device) for k, v in target.items()}
        model = SpritesDecomposer(
            target=target,
            prompt=prompt,
            cfg=cfg.model,
            out_dir=out_dir,
            device=cfg.device,
            resume=cfg.engine.resume,
        )
        model.fit(
            n_iters=cfg.engine.n_iters,
            max_time=timelimit_minute * 60 if timelimit_minute is not None else None,
            eval_interval=cfg.engine.eval_interval,
            save_interval=cfg.engine.save_interval,
            metric_fn=metric_fn,
            verbose=cfg.engine.verbose,
        )
        # eval
        output = {"frames": model.render(), "layers": model.render_layers()}
        output_np = {k: v.detach().cpu().numpy() for k, v in output.items()}
        target_np = {k: v.detach().cpu().numpy() for k, v in target.items()}
        metric = metric_fn(output_np, target_np)
        save_json(metric, osp.join(out_dir, "metric.json"))
        logger.info(metric)
