# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import torch

from .utils.latent_mask import _align_frame_count, _video_latent_t


class VLLMOmniH3TemporalMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.01}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 15.0}),
                "mode": (["continuation", "extension"],),
                "preserve_fraction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "IMAGE", "MASK")
    RETURN_NAMES = ("mask", "summary", "preview_fps", "preview_images", "preview_mask")
    FUNCTION = "build"
    CATEGORY = "vLLM-Omni"

    def build(self, images, source_fps, duration, mode, preserve_fraction=0.5):
        if images.shape[0] <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Source must contain frames and have a positive finite FPS.")
        if not math.isfinite(duration) or duration <= 0 or not 0 <= preserve_fraction <= 1:
            raise ValueError("Invalid duration or preserve_fraction.")
        frames = _align_frame_count(max(1, round(duration * 24)))
        source_seconds = images.shape[0] / source_fps
        if mode == "extension":
            if frames / 24 <= source_seconds:
                raise ValueError("Extension output must be longer than the source; increase duration.")
            boundary = source_seconds
        elif mode == "continuation":
            boundary = min(source_seconds, frames / 24) * preserve_fraction
        else:
            raise ValueError(f"Unknown temporal mask mode: {mode}")
        available = min(frames, math.floor(boundary * 24 + 1e-8))
        prefix = 0 if available < 5 else 5 + 17 * ((available - 5) // 17)
        preserved = _video_latent_t(prefix) if prefix else 0
        total = _video_latent_t(frames)
        mask = torch.ones(total, 1, 1)
        mask[:preserved] = 0
        indices = torch.arange(frames, device=images.device)
        source_indices = (indices * (source_fps / 24)).floor().long().clamp(max=images.shape[0] - 1)
        preview_images = images.index_select(0, source_indices)
        preview_mask = mask.index_select(0, torch.arange(frames) * total // frames)
        summary = (
            f"Source: {source_seconds:.3f}s; target: {frames} frames / {total} latent slices; "
            f"preserve: {preserved}; generate: {total - preserved}; "
            f"requested boundary: {boundary:.3f}s; conservative prefix: {prefix / 24:.3f}s."
        )
        return mask, summary, 24.0, preview_images, preview_mask
