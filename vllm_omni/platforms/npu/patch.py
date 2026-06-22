# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""General (non-310P) NPU monkeypatches for vLLM-Omni."""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRANSFORMERS_CAPTURE_PATCHED = False


def apply_npu_transformers_patches() -> None:
    """Make HF transformers' stream-capture detection aware of Ascend NPU aclgraph.

    transformers' ``is_tracing`` (used by ``create_causal_mask`` ->
    ``find_packed_sequence_indices``) guards a data-dependent host readback,
    ``(packed_sequence_mask[:, -1] == 0).all()``, behind ``is_cuda_stream_capturing()``.
    That helper only knows about CUDA stream capture, so under Ascend aclgraph
    capture the ``.all()`` still runs, forcing ``aclrtSynchronizeStream`` on a
    captured stream -> EZ1001 / error 107027 (e.g. capturing Qwen3-Omni code2wav,
    whose ``pre_transformer`` is a HF module).

    We OR-in ``torch.npu.is_current_stream_capturing()`` so ``is_tracing`` reports
    True during NPU capture and HF takes its static, capture-safe mask path.
    Outside capture the NPU check is False, so behavior and performance are
    unchanged. Both ``is_tracing`` and ``is_cuda_stream_capturing`` are defined in
    ``transformers.utils.import_utils``; ``is_tracing`` resolves the name from that
    module's globals at call time, so patching it there is what takes effect.
    """
    global _TRANSFORMERS_CAPTURE_PATCHED
    if _TRANSFORMERS_CAPTURE_PATCHED:
        return
    try:
        import torch
        import transformers.utils.import_utils as tf_import_utils
    except Exception:  # transformers/torch layout changed; nothing to patch
        return

    orig_is_cuda_stream_capturing = getattr(tf_import_utils, "is_cuda_stream_capturing", None)
    if orig_is_cuda_stream_capturing is None:
        return

    def _is_accel_stream_capturing() -> bool:
        try:
            if orig_is_cuda_stream_capturing():
                return True
        except Exception:
            pass
        try:
            return bool(torch.npu.is_current_stream_capturing())
        except Exception:
            return False

    tf_import_utils.is_cuda_stream_capturing = _is_accel_stream_capturing
    _TRANSFORMERS_CAPTURE_PATCHED = True
    logger.info_once(
        "Patched transformers.masking_utils.is_cuda_stream_capturing to detect "
        "Ascend NPU aclgraph capture (avoids host-sync in create_causal_mask)."
    )
