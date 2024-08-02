# Original source: https://github.com/gaomingqi/Track-Anything/blob/master/tools/painter.py

# paint masks, contours, or points on images, with specified colors

import cv2
import numpy as np
from PIL import Image


def colormap(rgb=True):
    color_list = np.array(
        [
            0.000,
            0.000,
            0.000,
            1.000,
            1.000,
            1.000,
            1.000,
            0.498,
            0.313,
            0.392,
            0.581,
            0.929,
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


color_list = colormap()
color_list = color_list.astype("uint8").tolist()


def vis_add_mask(image, mask, color, alpha):
    color = np.array(color_list[color])
    mask = mask > 0.5
    image[mask] = image[mask] * (1 - alpha) + color * alpha
    return image.astype("uint8")


def point_painter(
    input_image, input_points, point_color=5, point_alpha=0.9, point_radius=15, contour_color=2, contour_width=5
):
    h, w = input_image.shape[:2]
    point_mask = np.zeros((h, w)).astype("uint8")
    for point in input_points:
        point_mask[point[1], point[0]] = 1

    kernel = cv2.getStructuringElement(2, (point_radius, point_radius))
    point_mask = cv2.dilate(point_mask, kernel)

    contour_radius = (contour_width - 1) // 2
    dist_transform_fore = cv2.distanceTransform(point_mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1 - point_mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back
    # ...:::!!!:::...
    contour_radius += 2
    contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
    contour_mask = contour_mask / np.max(contour_mask)
    contour_mask[contour_mask > 0.5] = 1.0

    # paint mask
    painted_image = vis_add_mask(input_image.copy(), point_mask, point_color, point_alpha)
    # paint contour
    painted_image = vis_add_mask(painted_image.copy(), 1 - contour_mask, contour_color, 1)
    return painted_image


def mask_painter(input_image, input_mask, mask_color=5, mask_alpha=0.7, contour_color=1, contour_width=3):
    assert input_image.shape[:2] == input_mask.shape, "different shape between image and mask"
    # 0: background, 1: foreground
    mask = np.clip(input_mask, 0, 1)
    contour_radius = (contour_width - 1) // 2

    dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back
    # ...:::!!!:::...
    contour_radius += 2
    contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
    contour_mask = contour_mask / np.max(contour_mask)
    contour_mask[contour_mask > 0.5] = 1.0

    # paint mask
    painted_image = vis_add_mask(input_image.copy(), mask.copy(), mask_color, mask_alpha)
    # paint contour
    painted_image = vis_add_mask(painted_image.copy(), 1 - contour_mask, contour_color, 1)

    return painted_image


def background_remover(input_image, input_mask):
    """
    input_image: H, W, 3, np.array
    input_mask: H, W, np.array

    image_wo_background: PIL.Image
    """
    assert input_image.shape[:2] == input_mask.shape, "different shape between image and mask"
    # 0: background, 1: foreground
    mask = np.expand_dims(np.clip(input_mask, 0, 1), axis=2) * 255
    image_wo_background = np.concatenate([input_image, mask], axis=2)  # H, W, 4
    image_wo_background = Image.fromarray(image_wo_background).convert("RGBA")

    return image_wo_background
