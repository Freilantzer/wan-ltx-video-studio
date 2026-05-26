# Read-Only Comfy Reference Install

Date: 2026-05-26

Reference install inspected read-only:

```text
D:\IMAGE_GENERATORS\Comfy_UI_Furkan_V61\ComfyUI
```

No files were changed in that install. The notes below are for app design and runtime dependency planning.

## Core Runtime

- ComfyUI version: 0.17.0
- Python: 3.10.11
- Python executable: `D:\IMAGE_GENERATORS\Comfy_UI_Furkan_V61\ComfyUI\venv\Scripts\python.exe`
- Latest observed NVIDIA driver in log: 591.86
- Startup log reports RTX 5090 and about 30.15 GB free / 31.84 GB total CUDA memory at startup.
- Startup log reports SageAttention, Flash Attention, and Triton checks passing through the SeedVR2 optimization check.
- Startup log reports a Conv3D workaround for PyTorch 2.9.1 / cuDNN 91200 to reduce VAE memory pressure.

## Relevant Custom Nodes

### `ComfyUI-PainterLongVideo`

- Repository: `https://github.com/princepainter/ComfyUI-PainterLongVideo`
- Local commit: `889b4ff67909561e52d6ae023f5b9e8c33fdba94`
- Node: `PainterLongVideo`

This is the most important reference for the app's long-video behavior.

Useful behavior:

- Accepts `previous_video`, `start_image`, and `end_image` inputs.
- Uses the previous segment's last frame as the next segment's start point.
- Builds a gray latent sequence with protected/conditioned boundary frames.
- Adds `reference_motion` from recent previous frames.
- Adds `reference_latents`, including last-frame and optional initial-reference latents.
- Exposes `motion_frames` and `motion_amplitude`.

App implication: model this as a typed continuity strategy, not as a visible node.

### `comfyui-kjnodes`

- Installed through ComfyUI Manager/tracking, no `.git` directory present.
- Relevant node: `PathchSageAttentionKJ`

Useful behavior:

- Patches the model's `optimized_attention_override` with SageAttention.
- Can be applied per model in the workflow instead of relying only on a global Comfy launch flag.

App implication: attention mode should be part of the engine profile. Per-model attention patching may be safer than broad `--highvram` assumptions.

### `comfyui-videohelpersuite`

- Installed through ComfyUI Manager/tracking, no `.git` directory present.
- Relevant nodes:
  - `VHS_MergeImages`
  - `VHS_VideoCombine`

Useful behavior:

- `VHS_MergeImages` concatenates image/frame batches and can match dimensions.
- `VHS_VideoCombine` writes image sequences to video formats such as H.264 MP4.

App implication: the product should own segment concatenation and output metadata, whether it uses VHS underneath or a direct FFmpeg path.

### `ComfyUI-WanVideoWrapper`

- Installed through ComfyUI Manager/tracking, no `.git` directory present.

This was not the main node path in the inspected long-video workflow, but it is important future reference for optimized WAN execution.

Relevant features found in source:

- WAN model loading with attention mode options including `sdpa`, FlashAttention, SageAttention variants, and Comfy attention.
- Quantization options including FP8 scaled variants.
- Block swap controls for large models.
- VRAM management and auto CPU offload controls.
- LoRA selection with low-memory load and merge/unmerged options.
- Sampler `force_offload` behavior.
- Cache methods: TeaCache, MagCache, EasyCache.
- VAE loader option for CPU cache.

App implication: this is a strong reference for the app's advanced memory profile design, even if MVP starts with native Comfy WAN nodes.

### Future LTX Reference

- `ComfyUI-LTXVideo`
- Repository: `https://github.com/Lightricks/ComfyUI-LTXVideo`
- Local commit: `531512f7286963dc7aff1fd8bf5556e95eae03af`

App implication: keep LTX as a provider with its own constraints and controls, not as WAN-shaped settings.

### Other Optimization References

- `ComfyUI-GGUF`, commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
- `ComfyUI-TeaCache`, commit `81784223292fc5587fa6d785411332c5b3146a56`
- `ComfyUI-QuantOps`, commit `7ac371e1ad8536d5ff47f9e951e45790670084b3`

App implication: GGUF, caching, and quantized kernels should be treated as selectable engine capabilities after baseline chunked rendering works.

## Design Takeaways

- The product should not launch against this reference install directly.
- The reference workflow's node logic should become a typed `SegmentPlan` and `ContinuityPlan`.
- The app should generate or execute chunked jobs, not one monolithic long latent by default.
- The app should track per-chunk VRAM and timing.
- The app should trim duplicate boundary frames before final concat.
- The app should expose pixel-budget validation rather than hard-coding 720p.
- The app should preserve a distinction between:
  - default chunked WAN generation,
  - A14B pressure tests,
  - GGUF low-memory profiles,
  - cache/TeaCache experimental speed profiles,
  - future LTX provider workflows.
