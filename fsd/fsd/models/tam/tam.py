# Original source: https://github.com/gaomingqi/Track-Anything/blob/master/track_anything.py


import numpy as np

from .sam_controler import SamControler
from .tracker.base_tracker import BaseTracker


class TrackingAnything:

    def __init__(self, sam_checkpoint, xmem_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        # self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, config_path=args.xmem_config_path, device=args.device)
        # self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)

    # def inference_step(self, first_flag: bool, interact_flag: bool, image: np.ndarray,
    #                    same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     if first_flag:
    #         mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
    #         return mask, logit, painted_image

    #     if interact_flag:
    #         mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #         return mask, logit, painted_image

    #     mask, logit, painted_image = self.xmem.track(image, logit)
    #     return mask, logit, painted_image

    def first_frame_click(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    def first_frame_box(self, image: np.ndarray, box: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_box(image, box, multimask)
        return mask, logit, painted_image

    # def interact(self, image: np.ndarray, same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #     return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):

        masks = []
        logits = []
        painted_images = []
        for i in range(len(images)):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
