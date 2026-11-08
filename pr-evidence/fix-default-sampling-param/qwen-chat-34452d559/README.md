# Qwen image chat at 34452d559

- Source: `34452d559ba963a9ffc9e0e7660073704a33b563`; all 2299 production Python/YAML/Jinja source files verified against Git blobs.
- Same A100 / vLLM 0.33.0 environment, Qwen/Qwen-Image-2.1-Turbo weights, CLI defaults (8 steps, true CFG 1), 512x512 request and automatic-seed fixture as the previous run.
- The original edits request precedes chat, preserving the request order. Chat sends no sampling or seed fields; only test-harness automatic seed draws are fixed at 42.
- Chat: HTTP 200, **1.115 s**. API sampling: 8 steps / true CFG 1.
- Pixels match both main `2b610b3e9` and prior head `890eec92f`: MSE 0, PSNR infinity. This is an output-equivalence check for this request, not a performance or general quality claim.

See comparison.json, source-provenance.json and results/qwen/after for the request, sampling trace, launch arguments and output. The scripts are external validation fixtures; production code is unchanged.
