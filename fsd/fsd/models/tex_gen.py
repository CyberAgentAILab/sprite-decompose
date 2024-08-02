# Original source: https://github.com/vye16/deformable-sprites/blob/main/models/tex_gen.py
import torch
import torch.nn as nn

from .unet import UNet


def resample_batch(src, coords, **kwargs):
    """
    does 2D grid sample but for tensors of dimensions > 4
    """
    *sdims, C, H, W = src.shape
    *cdims, h, w, d = coords.shape
    assert sdims == cdims
    src = src.view(-1, C, H, W)
    coords = coords.view(-1, h, w, d)
    out = F.grid_sample(src, coords, **kwargs)
    return out.view(*sdims, C, h, w)


def resample_textures(texs, coords, random_comp=False):
    """
    :param texs (M, C, H, W)
    :param coords (b, m, h, w, 2)
    """
    out = {}
    M, C, H, W = texs.shape
    b, m, h, w, _ = coords.shape
    tex_rep = texs[None].repeat(b, 1, 1, 1, 1)  # (b, M, 3, H, W)
    apprs = resample_batch(tex_rep, coords, align_corners=False)
    rgb_apprs = get_rgb_layers(apprs, random_comp=random_comp)

    out["raw_apprs"] = apprs
    out["apprs"] = rgb_apprs
    return out


def get_rgb_layers(apprs, random_comp=False):
    """
    :param apprs (B, M, C, H, W)
    """
    B, M, C, H, W = apprs.shape
    if not random_comp:
        return apprs if C == 3 else apprs[:, :, :3] * apprs[:, :, 3:4]

    if C == 3:
        lo = 1e-3 * torch.rand(M, device=apprs.device).view(1, -1, 1, 1, 1)
        hi = 1 - 1e-3 * torch.rand(M, device=apprs.device).view(1, -1, 1, 1, 1)
        avg = apprs.detach().mean(dim=2, keepdim=True)
        maxed = (avg < lo) | (avg > hi)  # (B, M, 1, H, W)
        return ~maxed * apprs + maxed * torch.rand_like(apprs)

    rgb, fac = apprs[:, :, :3], apprs[:, :, 3:4]
    return fac * rgb + (1 - fac) * torch.rand_like(rgb)


class TexUNet(nn.Module):
    def __init__(
        self,
        n_layers,
        target_shape,
        n_channels=4,
        temperature=[1, 1, 1, 1],
        n_levels=1,
        d_code=4,
        d_hidden=16,
        fac=2,
        norm_fn="batch",
        random_comp=True,
        **kwargs
    ):
        super().__init__()

        # self.d_code = d_hidden // 2
        self.d_code = d_code
        self.n_layers = n_layers
        self.random_comp = random_comp
        assert len(temperature) == n_channels
        self.temperature = temperature

        tex_init = torch.rand(1, n_layers, self.d_code, *target_shape)
        # print("texture code shape", tex_init.shape)
        self.register_parameter("codes", nn.Parameter(tex_init, requires_grad=False))

        self.blocks = UNet(
            self.d_code,
            n_channels,
            n_levels=n_levels,
            d_hidden=d_hidden,
            fac=fac,
            norm_fn=norm_fn,
        )

    def forward(self, coords=None, vis=False):
        """
        returns the per-layer textures, optionally resamples according to provided coords
        :param coords (B, M, H, W, 2) optional
        :returns apprs (B, M, 3, H, W), texs (M, 3, H, W)
        """
        x = self.codes[0]  # (M, D, H, W)
        z = self.blocks(x)
        z = z * torch.tensor(self.temperature).to(x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # texs = torch.sigmoid(z)  # (M, 3, H, W)
        texs = z  # (M, 3, H, W)
        out = {"texs": texs[None]}  # (1, M, 3, H, W)

        if coords is not None:
            random_comp = self.random_comp and not vis
            out.update(resample_textures(texs, coords, random_comp))

        return out
