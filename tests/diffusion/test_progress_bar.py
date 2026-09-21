# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.progress_bar import (
    DiffusionProgress,
    ProgressBarMixin,
    progress_requests,
    progress_sink,
)


def request(identity, enabled=True):
    return SimpleNamespace(request_id=identity, sampling_params=SimpleNamespace(emit_request_lifecycle=enabled))


@pytest.mark.parametrize("disabled", [True, False])
def test_reports_steps_even_when_terminal_bar_disabled(disabled):
    pipeline = ProgressBarMixin()
    pipeline.set_progress_bar_config(disable=disabled)
    events = []
    with progress_sink(events.append), progress_requests([request("a"), request("b"), request("offline", False)]):
        with pipeline.progress_bar(total=3) as bar:
            for _ in range(3):
                bar.update()
    assert events == [DiffusionProgress(rid, step, 3) for step in range(1, 4) for rid in ("a", "b")]


def test_request_context_does_not_leak_after_failure():
    pipeline = ProgressBarMixin()
    pipeline.set_progress_bar_config(disable=True)
    events = []
    with progress_sink(events.append):
        with pytest.raises(RuntimeError), progress_requests([request("failed")]):
            with pipeline.progress_bar(total=2) as bar:
                bar.update()
                raise RuntimeError("generation failed")
        with pipeline.progress_bar(total=2) as bar:
            bar.update()
        with progress_requests([request("next")]), pipeline.progress_bar(total=1) as bar:
            bar.update()
    assert events == [DiffusionProgress("failed", 1, 2), DiffusionProgress("next", 1, 1)]
