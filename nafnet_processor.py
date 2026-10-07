"""
NAFNet SOTA Image Restoration & Deblurring Engine (Pure PyTorch FP16 + CUDA).

Provides state-of-the-art non-linear activation free image deblurring, denoising,
and clarity enhancement. Uses tiled spatial processing to prevent GPU VRAM
exhaustion and float16 mixed precision for near-instant inference.
Zero external dependencies beyond PyTorch.
"""

import os
import gc
import math
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger("nafnet")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_BASE_DIR, "models", "NAFNet-GoPro-width64.pth")


def is_available():
    """Return True if weights exist on disk and PyTorch is available."""
    return os.path.exists(_WEIGHTS_PATH)


# ── NAFNet ARCHITECTURE ───────────────────────────────────────────────────────

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)

        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


def _load_model(device):
    model = NAFNet(img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    checkpoint = torch.load(_WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16).eval()
    else:
        model = model.to(device=device, dtype=torch.float32).eval()
    return model


def enhance_clarity_gpu(src_path, out_path, tile_size=512, tile_pad=32):
    """
    Restores photo sharpness, clarity, and removes motion/lens blur using NAFNet SOTA.
    Uses bfloat16/float32 to prevent numerical overflow and tile-based inference
    to conserve VRAM on 6 GB RTX 5050.
    """
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {src_path}")
    h, w = img.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = _load_model(device)

    try:
        # Preprocessing: BGR -> RGB, [0, 1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=dtype)

        # If image fits in tile_size, run directly
        if h <= tile_size and w <= tile_size:
            with torch.no_grad():
                out_tensor = model(tensor)
        else:
            # Tiled inference
            out_tensor = torch.zeros_like(tensor)
            h_steps = math.ceil(h / tile_size)
            w_steps = math.ceil(w / tile_size)

            with torch.no_grad():
                for hi in range(h_steps):
                    for wi in range(w_steps):
                        top = hi * tile_size
                        left = wi * tile_size
                        bottom = min(top + tile_size, h)
                        right = min(left + tile_size, w)

                        top_pad = max(top - tile_pad, 0)
                        left_pad = max(left - tile_pad, 0)
                        bottom_pad = min(bottom + tile_pad, h)
                        right_pad = min(right + tile_pad, w)

                        tile = tensor[:, :, top_pad:bottom_pad, left_pad:right_pad]
                        out_tile = model(tile)

                        # Extract unpadded region
                        crop_top = top - top_pad
                        crop_left = left - left_pad
                        crop_bottom = crop_top + (bottom - top)
                        crop_right = crop_left + (right - left)

                        out_tensor[:, :, top:bottom, left:right] = out_tile[:, :, crop_top:crop_bottom, crop_left:crop_right]

        out_arr = out_tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()
        out_bgr = cv2.cvtColor((out_arr * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, out_bgr)
        return {"status": "success", "engine": "nafnet_sota_gpu"}
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
