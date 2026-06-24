# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Monkey-patch ``Qwen3OmniMoeCode2Wav.__init__`` for NPU Code2Wav runtime knobs."""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_init = None


def _prepare_npu_code2wav_runtime() -> None:
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_npu():
        return
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


def _patched_init(self, *, vllm_config: VllmConfig | None = None, prefix: str = "") -> None:
    _prepare_npu_code2wav_runtime()
    assert _original_init is not None
    _original_init(self, vllm_config=vllm_config, prefix=prefix)


def _npu_pre_transformer_attention_mask(self, inputs_embeds):
    """
    NPU override: build the ``pre_transformer`` mask in an aclgraph-capture-safe way.

    HF's auto path routes into ``find_packed_sequence_indices``, whose
    ``(packed_sequence_mask[:, -1] == 0).all()`` is a host readback guarded only by
    ``is_cuda_stream_capturing()`` -- which cannot see NPU aclgraph capture, so it syncs
    the captured stream -> EZ1001. code2wav decodes a single contiguous codec sequence
    (never packed), so we build the masks with ``position_ids=None`` to skip that
    detection while producing the same causal / sliding-window masks.

    Args:
        inputs_embeds: [batch, seq_len, hidden_size] - code-embedding output

    Returns:
        attention_mask: {attention_type: 4D mask} mapping reused by pre_transformer verbatim
    """
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    pre = self.pre_transformer
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    mask_kwargs = {
        "config": pre.config,
        "input_embeds": inputs_embeds,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": None,  # skip find_packed_sequence_indices host-sync
    }
    mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if pre.has_sliding_layers:
        mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return mask_mapping


def apply_qwen3_omni_code2wav_patch() -> None:
    global _PATCHED, _original_init
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav

    _original_init = Qwen3OmniMoeCode2Wav.__init__
    Qwen3OmniMoeCode2Wav.__init__ = _patched_init  # type: ignore[method-assign]
    # Override the platform-agnostic no-op hook with the capture-safe NPU builder.
    Qwen3OmniMoeCode2Wav._pre_transformer_attention_mask = _npu_pre_transformer_attention_mask  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for Qwen3OmniMoeCode2Wav.__init__ / _pre_transformer_attention_mask")
