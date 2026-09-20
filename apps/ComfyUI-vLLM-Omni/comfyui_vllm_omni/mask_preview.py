# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.nn.functional as F

from .utils.latent_mask import video_mask_to_grid


class VLLMOmniH3MaskGridPreview:
    """Overlay the request's static spatial mask grid on source frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "opacity": ("FLOAT", {"default": 0.48, "min": 0.0, "max": 1.0, "step": 0.01}),
                "show_grid": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preview"
    CATEGORY = "vLLM-Omni/Video"
    DESCRIPTION = (
        "Static H3 spatial mask preview: red means regenerate. Uses input video dimensions "
        "rounded down to multiples of 32; match generation dimensions before previewing. "
        "Preserves source frame count and FPS. Temporal masks are not supported."
    )

    def preview(self, images, mask, opacity=0.48, show_grid=True):
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 3:
            raise ValueError("Expected a nonempty IMAGE batch with RGB channels.")
        if mask.ndim == 3 and mask.shape[0] > 1:
            if not torch.equal(mask, mask[:1].expand_as(mask)):
                raise ValueError("H3 Mask Grid Preview supports static masks only, not temporal masks.")
            mask = mask[:1]
        if not torch.isfinite(mask).all() or mask.numel() == 0 or mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask values must be finite and in [0, 1].")
        height, width = (int(v) // 32 * 32 for v in images.shape[1:3])
        if min(height, width) < 32:
            raise ValueError("H3 preview requires width and height of at least 32.")
        grid = video_mask_to_grid(mask, width=width, height=height, num_frames=5)[0]
        grid = grid.to(device=images.device, dtype=images.dtype)
        cells = grid.repeat_interleave(16, 0).repeat_interleave(16, 1).unsqueeze(-1)
        alpha = cells * opacity
        lines = torch.zeros_like(cells)
        if show_grid:
            lines[::16] = 0.16
            lines[:, ::16] = 0.16
            lines = torch.where((lines > 0) & (cells > 0), 0.55, lines)
        red = images.new_tensor([1.0, 32 / 255, 32 / 255])
        yellow = images.new_tensor([1.0, 220 / 255, 100 / 255])
        result = images.new_empty((images.shape[0], height, width, 3))
        # Work one frame at a time to avoid multiple full-video temporary tensors.
        for index, frame in enumerate(images):
            frame = frame[..., :3]
            if frame.shape[:2] != (height, width):
                frame = F.interpolate(
                    frame.permute(2, 0, 1)[None], size=(height, width), mode="bilinear", align_corners=False
                )[0].permute(1, 2, 0)
            result[index] = (frame * (1 - alpha) + red * alpha) * (1 - lines) + yellow * lines
        return (result,)
