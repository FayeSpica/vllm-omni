# H3 latent-mask editing (WF-05)

Open `vLLM-Omni H3 Latent Editing` from the extension templates. It contains four independent groups: inpainting, object removal, continuation, and extension. Only inpainting is enabled initially. Set the current group's nodes to **Never**, then enable the group you want to run; do not use Bypass for disabling these examples.

Requires EDIT-01 server support (#7465), EDIT-02 nodes (#7575), and the audio-preserving ComfyUI video output path. Both EDIT PRs are still open as of September 18, 2026. This is a workflow draft, not a claim that stock main supports latent editing. Set the URL and served model name in the selected Generate Video node. The template defaults to `http://127.0.0.1:8001/v1` and `MiniMaxAI/MiniMax-H3`.

Upload a 1344 × 768 source clip with **107 frames at 24 FPS**. File selectors intentionally start empty and contain no machine-specific paths. All outputs use 24 FPS and a native 1344 × 768 resolution. Keep the Save Video connection to preserve the returned soundtrack.

- **Inpainting:** upload a black/white mask using the red channel and describe the replacement. White regenerates; black preserves. The spatial mask repeats over time.
- **Object removal:** upload a mask covering the unwanted object's complete motion path and describe the background to reconstruct. A static mask is unsuitable for precise tracking; connect a tracked mask batch if needed.
- **Continuation:** retain the beginning and generate a new ending at the same 107-frame duration. The mask contains 16 preserved and 16 regenerated latent time slices.
- **Extension:** extend the 107-frame source to 209 frames. The mask contains 32 preserved and 30 regenerated latent time slices. EDIT-01 pads the source with its last frame before encoding; the generated tail replaces that padded region.

At 24 FPS, durations `4.458` and `8.708` round to 107 and 209 frames. Both satisfy `17k+5`. Their latent time lengths are `2 + 5 * ((frames - 5) / 17)`, giving 32 and 62. The temporal mask uses these latent lengths directly to avoid resampling its boundary. If you change the source or output length, update the mask counts too. Preservation operates on encoded latents and does not promise a lossless pixel splice.

Inpainting and removal use `audio_mask=0` and require source audio. For a silent source, set it to `1` to generate audio. Continuation and extension use `audio_mask=1`, regenerating the entire soundtrack. The current EDIT-02 node exposes only a scalar audio mask; retaining the original audio prefix while generating only its tail is not represented by this template.

Validation command:

```bash
python -m pytest --noconftest -o addopts='' tests/e2e/features/comfyui/test_wf05_template.py -q
```

This graph-only check does not need engine fixtures or distributed pytest options. The focused tests check wiring, activation defaults, frame constraints, and temporal mask construction. Local result: **4 passed**; node types and connected port names also match the running ComfyUI's `/object_info`. Targeted pre-commit checks passed. Real-model validation is pending for all four groups. Before claiming WF-05 complete, record each input/mask/output, confirm dimensions, frame count and audio presence, and review preserved regions, edited regions, temporal continuity and audio synchronization.
