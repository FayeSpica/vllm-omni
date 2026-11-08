# Before / after: requests omit sampling fields

- Before: main `2b610b3e9`; after: PR `890eec92f`.
- Same vLLM 0.33.0 environment, model weights, startup arguments and request bodies.
- HTTP requests contain no steps, CFG, shift, seed or generator-device fields.
- Test-only automatic-seed draws return 42 in both servers. The harness records API-to-engine and backend sampling values; it does not set steps/CFG/shift or mock model generation.
- All requests returned HTTP 200. Latencies are single observations, not benchmarks.

| API | Before / after (s) | Result |
| --- | --- | --- |
| MOSS image generation | 14.056 / 27.562 | CFG 1 -> 3; expected image change |
| Qwen 2.1 Turbo edits | 2.029 / 2.015 | Pixel-identical |
| Qwen 2.1 Turbo image chat | 1.120 / 1.082 | Pixel-identical; chat inheritance corrected |
| H3 videos | 139.605 / 137.923 | Video pixels and audio PCM identical |
| H3 videos/sync | 133.968 / 134.005 | Video pixels and audio PCM identical |

MOSS uses a plain startup without CLI sampling defaults. Before handed None/None/{} to the engine (steps/CFG/extras), then CFG normalized to 1; after hands 50/3/{flow_shift:5}. The checkpoint scheduler itself also defaults to shift 5. The MOSS pixel change confirms changed behavior, not a general-purpose quality score.

Qwen keeps its existing generic configuration path; no Qwen YAML or pipeline registration is added. Qwen uses deployed CLI defaults of 8 steps and true CFG 1. Edits already inherited them on main. Before chat handed None/None for steps/true CFG; after hands 8/1. The checkpoint's eight sample_sigmas and true-CFG fallback 1 make both default chat outputs identical. This does not prove arbitrary CLI overrides were inherited before.

H3 uses its default 50-step schedule and shifts 12/3, with layerwise offload for DiT and text encoder. 100 requested frames at 24 fps align to 107 decoded frames, 512x288, with non-silent 32 kHz audio. These are bounded cases, not a claim of no regression for every request.

Qwen edit reference: reference.png, RGB SHA-256 70499c16dea54bddb1d67831f2df11bc5e0a45ec01b2b99ef2bc3a2d532bf532. Exact request bodies, sampler traces and results are stored per model/version. serve.py and test.py are the external validation harness, not production code changes.

API/config regression at `890eec92f`: 1447 passed, 1 skipped, 1 xfailed in 51.86 s. See unit-test.log. Audio-generation model E2E remains deferred.
