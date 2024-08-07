# import for debugging

import numpy as np

# import progressbar
# import for base_tracker
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from ..painter import mask_painter
from .inference.inference_core import InferenceCore
from .model.network import XMem
from .util.mask_mapper import MaskMapper
from .util.range_transform import im_normalization


class BaseTracker:

    def __init__(
        self, xmem_checkpoint, device, config_path="tracker/config/config.yaml", sam_model=None, model_type=None
    ) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        # initialise XMem
        network = XMem(config, xmem_checkpoint, map_location=device).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.device = device

        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

        # # SAM-based refinement
        # self.sam_model = sam_model
        # self.resizer = Resize([256, 256])

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h / min_hw * self.size), int(w / min_hw * self.size)), mode="nearest")

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input:
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:  # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None
        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        # track one frame
        probs, _ = self.tracker.step(frame_tensor, mask, labels)  # logits 2 (bg fg) H W
        # # refine
        # if first_frame_annotation is None:
        #     out_mask = self.sam_refinement(frame, logits[1], ti)

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)

        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs + 1):
            if np.max(final_mask == obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (final_mask == obj).astype("uint8"), mask_color=obj + 1)

        # print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')

        return final_mask, final_mask, painted_image

    @torch.no_grad()
    def sam_refinement(self, frame, logits, ti):
        """
        refine segmentation results with mask prompt
        """
        # convert to 1, 256, 256
        self.sam_model.set_image(frame)
        mode = "mask"
        logits = logits.unsqueeze(0)
        logits = self.resizer(logits).cpu().numpy()
        prompts = {"mask_input": logits}  # 1 256 256
        masks, scores, logits = self.sam_model.predict(
            prompts, mode, multimask=True
        )  # masks (n, h, w), scores (n,), logits (n, 256, 256)
        painted_image = mask_painter(frame, masks[np.argmax(scores)].astype("uint8"), mask_alpha=0.8)
        painted_image = Image.fromarray(painted_image)
        painted_image.save(f"/ssd1/gaomingqi/refine/{ti:05d}.png")
        self.sam_model.reset_image()

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()
