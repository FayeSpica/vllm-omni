# H3 latent-mask editing (WF-05)

Open `vLLM-Omni MiniMax-H3 Latent Mask Editing`. Upload a source video. Each spatial case has an independent mask that is built with official Solid Mask and Combine Masks nodes; no mask image is required. Configure a compatible H3 server URL and model in each generation node. The workflow uses the EDIT-02 latent-edit interface.

Each case has a Mask Preview above its Result in the Output Hub. Select the desired Save Video output and execute only that output to preview locally without running generation. All four cases are enabled; running the whole graph also submits generation requests.

Set output dimensions and duration directly in Generate Video. For continuation and extension, also update the duration in MiniMax-H3 Temporal Mask. Defaults are 1344 x 768 at 24 FPS, with 5 seconds for removal, inpainting and continuation, and 10 seconds for extension.

Preview uses official ComfyUI nodes only. Spatial cases use Empty Image, Image Composite Masked and Image Blend to overlay the original mask in red on source frames, followed by Create Video and Save Video. This preview follows the source resolution, FPS and duration and does not visualize the model's downsampled latent mask. Temporal cases use the same official red-overlay chain. The temporal node samples at 24 FPS and holds the last source frame for extension. With a 24 FPS source, existing frames are selected directly. Preserved frames remain unchanged; generated regions receive a red tint. This is a schematic timeline, not exact decoded-frame boundaries. Neither preview runs inference; previews are silent.

MiniMax-H3 Temporal Mask calculates target frames and latent slice counts from the decoded source video. Continuation preserves the selected fraction of the shorter source/target duration. Extension preserves the source prefix and requires a longer target. The target rounds up to `17k+5` frames; preservation rounds down to complete native chunks. Set the same duration in this node and Generate Video. The node also supplies source frames and a schematic per-frame mask to the official red-overlay preview nodes.

Audio masks remain whole-clip scalars. Zero preserves encoded source audio, one regenerates it, and intermediate values partially edit the audio latents. Neither latent video nor audio preservation guarantees lossless reconstruction.

The exported workflow has an empty video selector. The default rectangle is x=1074, y=493, width=180, height=168 on a 1344 x 768 canvas, including 32 pixels of background margin around the original bag mask bounding box. Adjust Mask Canvas to the source dimensions, Mask Rectangle to the object size, and Combine Masks x/y to its position. A filled rectangle may edit more background than the original shaped mask. Provide a matching school clip or adapt the rectangle and prompts to your assets. Reopen the workflow without saving an older canvas over it.

Graph tests check wiring, matching preview/generation defaults, mask sources, vertical output placement and portable inputs. They do not establish real-model visual/audio acceptance.

The example prompts use wall reconstruction, a curled cat with a swaying tail, heavy rain with distant lightning, and a summer-night firework. Current audio masks are 0, 0, 0.8, and 0.9 respectively; Continuation preserves 0.2 of the shorter source/target duration.
