# Original source: https://github.com/gaomingqi/Track-Anything/blob/master/tools/interact_tools.py


import numpy as np
from PIL import Image

from .base_segmenter import BaseSegmenter
from .painter import mask_painter, point_painter

mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SamControler:

    def __init__(self, SAM_checkpoint, model_type, device):
        """
        initialize sam controler
        """

        self.sam_controler = BaseSegmenter(SAM_checkpoint, model_type, device)

    # def seg_again(self, image: np.ndarray):
    #     '''
    #     it is used when interact in video
    #     '''
    #     self.sam_controler.reset_image()
    #     self.sam_controler.set_image(image)
    #     return

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True, mask_color=3
    ):
        """
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        """
        self.sam_controler.set_image(image)
        # origal_image = self.sam_controler.orignal_image
        neg_flag = labels[-1]
        if neg_flag == 1:
            # find neg
            prompts = {
                "point_coords": points,
                "point_labels": labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, "point", multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            prompts = {"point_coords": points, "point_labels": labels, "mask_input": logit[None, :, :]}
            masks, scores, logits = self.sam_controler.predict(prompts, "both", multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:
            # find positive
            prompts = {
                "point_coords": points,
                "point_labels": labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, "point", multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        assert len(points) == len(labels)

        painted_image = mask_painter(image, mask.astype("uint8"), mask_color, mask_alpha, contour_color, contour_width)
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels > 0)], axis=1),
            point_color_ne,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels < 1)], axis=1),
            point_color_ps,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = Image.fromarray(painted_image)

        return mask, logit, painted_image

    def first_frame_box(self, image: np.ndarray, box: np.ndarray, multimask=True, mask_color=3, box_color=2):
        """
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        """
        self.sam_controler.set_image(image)
        # origal_image = self.sam_controler.orignal_image
        prompts = {"box": box}
        masks, scores, logits = self.sam_controler.predict(prompts, "box", multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        painted_image = mask_painter(image, mask.astype("uint8"), mask_color, mask_alpha, contour_color, contour_width)
        box_mask = np.zeros_like(mask)
        box_mask[int(box[0][1]) : int(box[0][3]), int(box[0][0]) : int(box[0][2])] = 1
        painted_image = mask_painter(
            painted_image, box_mask.astype("uint8"), box_color, 0, contour_color, contour_width
        )
        painted_image = Image.fromarray(painted_image)

        return mask, logit, painted_image
