# Original source: https://github.com/gaomingqi/Track-Anything/blob/e6e159273790974e04eeea6673f1f93c035005fc/app.py
import os
import os.path as osp

import fsd
import numpy as np
import requests
import torch
from easydict import EasyDict as edict
from fsd.utils.log import get_logger

from .tam.tam import TrackingAnything

logger = get_logger(name=__name__)
SAM_checkpoint_dict = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
SAM_checkpoint_url_dict = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def build_tam(sam_model_type="vit_b", device="cpu"):
    # check and download checkpoints if needed
    sam_checkpoint = SAM_checkpoint_dict[sam_model_type]
    sam_checkpoint_url = SAM_checkpoint_url_dict[sam_model_type]
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"

    folder = osp.join(fsd.__path__[0], "cache", "checkpoints")
    sam_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
    xmen_config_path = osp.join(fsd.__path__[0], "configs", "xmem", "xmem.yaml")
    args = edict(
        {
            "device": device,
            "sam_model_type": sam_model_type,
            "port": 6080,
            "debug": False,
            "mask_save": False,
            "xmem_config_path": xmen_config_path,
        }
    )
    tam = TrackingAnything(sam_checkpoint=sam_checkpoint, xmem_checkpoint=xmem_checkpoint, args=args)
    tam.samcontroler.sam_controler.reset_image()
    tam.xmem.clear_memory()
    return tam


def expand_box(box: np.ndarray, mergin: float = 0.1) -> np.ndarray:
    """
    box: (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = box
    h = y1 - y0
    w = x1 - x0
    y0 -= h * mergin
    y1 += h * mergin
    x0 -= w * mergin
    x1 += w * mergin
    return np.array([y0, y1, x0, x1])


def segmentation_tam_key_box_prompt(
    frames: np.ndarray,
    box: np.ndarray,
    key_index: int,
    require_unnormalize: bool = False,
    device: str = "cpu",
    mergin: float = 0.05,
) -> np.ndarray:
    """
    frames: Array with shape (T, H, W, C)
    box: Array with shape (4,) (y0, y1, x0, x1)
    require_unnormalize: If True, frames should be in [0, 1]
    device: "cpu" or "cuda"
    mergin: Mergin ratio for expanding the box
    >>>
    Array with shape (T, H, W)
    """
    assert not np.isnan(box).any(), f"box should not contain NaNs: {box}"
    frames = frames[..., :3]  # rgb
    if require_unnormalize:
        frames = (frames * 255).astype(np.uint8)
    t, h, w, c = frames.shape
    tam = build_tam(device=device)
    box = expand_box(box, mergin=mergin)
    box = box[[2, 0, 3, 1]] * np.array([w, h, w, h])  # left, top, right, bottom
    key_frame = frames[key_index]
    key_mask, _, _ = tam.first_frame_box(
        image=key_frame,
        box=np.array([box]),
    )
    if key_mask.mean() == 0:
        logger.warning("key_mask is empty")
        return np.zeros((t, h, w))

    key_index = key_index if key_index != -1 else len(frames) - 1
    # forward
    forward_masks, _, _ = tam.generator(frames[key_index:], key_mask)
    tam.xmem.clear_memory()
    # backward
    backward_masks, _, _ = tam.generator(frames[: key_index + 1][::-1], key_mask)

    masks = np.array(backward_masks[::-1][:-1] + [key_mask] + forward_masks[1:])
    del tam
    torch.cuda.empty_cache()
    return masks
