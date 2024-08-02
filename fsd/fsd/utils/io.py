import io
import json
from typing import List, Optional, Union

import av
import cv2
import fsspec
import imageio
import numpy as np
import torch
from apng import APNG, PNG
from PIL import Image


def read_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def read_image_pil(path: str) -> Image.Image:
    with fsspec.open(path, "rb") as f:
        image = Image.open(io.BytesIO(f.read()))
    return image


def save_image_pil(image: Union[Image.Image, np.ndarray], path: str) -> None:
    assert isinstance(image, (Image.Image, np.ndarray)), "frames must be Image.Image or np.ndarray"
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    with fsspec.open(path, "wb") as f:
        image.save(f, format="PNG")


def read_video(path: str, height: Optional[int] = None, require_alpha: bool = True) -> List[np.ndarray]:
    if path.endswith(".mp4"):
        return read_mp4(path)
    elif path.endswith((".gif", ".png")):
        return read_gif(path, height=height, require_alpha=require_alpha)


def read_mp4(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def read_gif(
    path: str, height: Optional[int] = None, require_alpha: bool = True, backend: str = "av"
) -> List[np.ndarray]:
    assert backend in ["av", "imageio", "pillow"]
    if backend == "av":
        return read_gif_av(path, height=height, require_alpha=require_alpha)
    elif backend == "imageio":
        return read_gif_imageio(path, height=height, require_alpha=require_alpha)
    elif backend == "pillow":
        return read_gif_pillow(path, height=height)


def read_gif_av(path: str, height: Optional[int] = None, require_alpha: bool = True) -> List[np.ndarray]:
    with fsspec.open(path, "rb") as f:
        c = av.open(f)
        frames = [f.to_ndarray(format="rgba") for f in c.decode(video=0)]
    if height is not None:
        codec_context = c.streams[0].codec_context
        scale = height / codec_context.height
        width = round(codec_context.width * scale)
        frames = [cv2.resize(f, dsize=(width, height)) for f in frames]
    if not require_alpha:
        frames = [f[..., :3] for f in frames]
    return frames


def read_gif_imageio(path: str, height: Optional[int] = None, require_alpha: bool = True) -> List[np.ndarray]:
    with fsspec.open(path, "rb") as f:
        reader = imageio.get_reader(f)
        frames = [f for f in reader]
    if height is not None:
        frames = [cv2.resize(f, dsize=(round(f.shape[1] * height / f.shape[0]), height)) for f in frames]
    if not require_alpha:
        frames = [f[..., :3] for f in frames]
    return frames


def read_gif_pillow(path: str, height: Optional[int] = None) -> List[Image.Image]:
    with fsspec.open(path, "rb") as f:
        apng = Image.open(f)
        frames = []
        for i in range(apng.n_frames):
            apng.seek(i)
            frames.append(apng.copy())
    if height is not None:
        frames = [f.resize((round(f.width * height / f.height), height)) for f in frames]
    return frames


def save_frames_as_apng(frames: List[Union[Image.Image, np.ndarray]], out_path: str, fps: int = 10) -> None:
    assert isinstance(frames[0], (Image.Image, np.ndarray)), "frames must be Image.Image or np.ndarray"
    if isinstance(frames[0], np.ndarray):
        frames = [Image.fromarray(f) for f in frames]
    delay = round((1.0 / fps) * 1000)
    apng = APNG()
    for i, f in enumerate(frames):
        f_io = io.BytesIO()
        f.save(f_io, format="PNG")
        png = PNG.from_bytes(f_io.getvalue())
        f_io.close()
        apng.append(png, delay=delay)
    with fsspec.open(out_path, "wb") as f:
        apng.save(f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            obj = int(obj)
        elif isinstance(obj, np.floating):
            obj = float(obj)
        elif isinstance(obj, np.ndarray):
            obj = obj.tolist()
        elif isinstance(obj, torch.Tensor):
            obj = obj.tolist()
        else:
            obj = super(NumpyEncoder, self).default(obj)
        if isinstance(obj, list) and (len(obj) > 0):
            if isinstance(obj[0], bytes):
                obj = [elem.decode() for elem in obj]
        return obj
