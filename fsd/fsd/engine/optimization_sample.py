import os.path as osp


import numpy as np
import torch as tr
from fsd.models.decomposer import SpritesDecomposer
from fsd.utils.io import read_json, read_video
from fsd.utils.log import get_logger
from omegaconf import DictConfig

logger = get_logger(name=__name__)


def optimization_sample(cfg: DictConfig) -> None:
    # Read video and prompt
    template_id = cfg.in_dir.split("/")[-1]
    frames = read_video(osp.join(cfg.in_dir, f"{template_id}.png"))
    prompt = read_json(osp.join(cfg.in_dir, f"{template_id}.json"))

    # Make inputs
    inputs = {"frames": tr.tensor(np.array(frames)).float().to(cfg.device) / 255.0}
    prompt = {k: tr.tensor(v).to("cpu") for k, v in prompt.items()}

    # Optimization
    model = SpritesDecomposer(
        target=inputs,
        prompt=prompt,
        cfg=cfg.model,
        out_dir=cfg.out_dir,
        device=cfg.device,
        resume=cfg.engine.resume,
    )
    model.fit(
        n_iters=cfg.engine.n_iters,
        max_time=cfg.engine.timelimit_minute * 60 if cfg.engine.timelimit_minute is not None else None,
        eval_interval=cfg.engine.eval_interval,
        save_interval=cfg.engine.save_interval,
        verbose=cfg.engine.verbose,
    )
