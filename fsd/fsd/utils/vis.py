import math
from typing import Tuple, Union

import numpy as np
from einops import repeat
from PIL import Image


def get_checker_board_image(
    size: Tuple[int, int], block_size: Tuple[int, int] = (10, 10), black_value: int = 160, white_value: int = 190
) -> Image.Image:
    """Generate a checkerboard image with a given size and block size.

    Args:
        size (Tuple[int, int]): The size of the image (width, height).
        block_size (Tuple[int, int]): The size of each block in the checkerboard (width, height). Defaults to (10, 10).
        black_value (int): The pixel value for black blocks. Defaults to 160.
        white_value (int): The pixel value for white blocks. Defaults to 190.

    Returns:
        Image.Image: The generated checkerboard image with RGBA channels.
    """
    size = (size[1], size[0])
    n_blocks = [math.ceil(size[0] / block_size[0]), math.ceil(size[1] / block_size[1])]
    row = np.zeros([n_blocks[1]]).astype(bool)
    row[::2] = True
    blocks = repeat(row, "w -> h w", h=n_blocks[0])
    blocks[::2] = np.logical_not(blocks[::2])

    blocks_int = blocks.astype(np.uint8)
    blocks_int[np.where(blocks)] = white_value
    blocks_int[np.where(np.logical_not(blocks))] = black_value

    cb_image = repeat(blocks_int, "h w -> (h bh) (w bw) c", bh=block_size[0], bw=block_size[1], c=3)
    cb_image = cb_image[: size[0], : size[1]]
    alpha = (np.ones([size[0], size[1], 1]) * 255).astype(np.uint8)
    cb_image = np.concatenate([cb_image, alpha], axis=2)
    cb_image = Image.fromarray(cb_image).convert("RGBA")

    return cb_image


def insert_checker_board_bg(
    image: Union[Image.Image, np.ndarray], block_size: Tuple[int, int] = (10, 10)
) -> Image.Image:
    """Insert a checkerboard background into an image.

    Args:
        image (Union[Image.Image, np.ndarray]): The image to insert the checkerboard background into.
        block_size (Tuple[int, int]): The size of each block in the checkerboard (width, height). Defaults to (10, 10).

    Returns:
        Image.Image: The image with the checkerboard background inserted.
    """
    image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    size = image.size
    block_size = block_size if isinstance(block_size, tuple) else (min(size) // 10, min(size) // 10)
    cb_image = get_checker_board_image(image.size, block_size)
    image = Image.alpha_composite(cb_image, image)
    return image
