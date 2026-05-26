# Direct Renderer Backend

The backend target is a standalone render server, not a ComfyUI wrapper. ComfyUI is only a reference for proven WAN 2.2 behavior on the target RTX 5090.

## Reference Contract

The known-good reference run is WAN 2.2 I2V A14B at 1280x720, 16 fps, three 81-frame input chunks, about 15.06 seconds of output, and roughly 25 GB dedicated VRAM. The backend treats that as the memory target for the A14B 720p profile:

- target: 25 GB
- warning: 28 GB
- unsafe: 30 GB

The direct renderer must keep the high-noise and low-noise A14B experts mutually exclusive on CUDA. The text encoder can be moved to CUDA for encoding and then returned to system RAM before diffusion, and VAE decode should happen after DiT experts are offloaded.

## Current Backend Slice

The first backend slice resolves local model files, builds render-job commands, and exposes memory policy. GPU work remains opt-in: the segment runner refuses to execute unless `--allow-gpu` is present.

Endpoints:

- `GET /api/render/profiles`
- `POST /api/render/plan`
- `POST /api/render/segment-command`

The render plan includes:

- selected WAN/LTX profile
- required model components and resolved paths
- built-in workflow LoRAs, such as Lightning profile components
- requested creative LoRAs
- segment commands with frame count, prompt, seed, trim, sample steps, and offload flags
- VRAM policy and allocator settings
- execution stages, with GPU execution marked pending

The first executable runner path is `wan22_ti2v_5b_fp16`. It is a smoke-test path for the direct render loop because its diffusion safetensors match the upstream WAN module keys directly. The A14B FP8 profiles stay planned-only until the backend has custom FP8 linear support for the scaled Comfy-style weights.

The single-segment runner lives at:

```text
python -m wan_ltx_studio.rendering.single_segment_runner
```

Safe dry-runs use `--dry-run` and do not import Torch or create a CUDA context. Real GPU execution requires `--allow-gpu`.

## Implementation Notes

The upstream WAN runtime already has a useful reference behavior in `WanI2V._prepare_model_for_timestep`: it chooses the active high/low expert by timestep and moves the inactive expert to CPU when offloading is enabled. The production backend should preserve that behavior, then adapt loading to our local Comfy-style safetensors and GGUF assets.

The current planned allocator matches the working reference script:

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
```

## First GPU Smoke Result

The first executable direct render completed on the RTX 5090 with `wan22_ti2v_5b_fp16`:

- command profile: `wan22_ti2v_5b_fp16`
- prompt-only TI2V smoke render, no start image
- size: `640x352`
- frames: `25`
- fps: `24`
- sampling: `2` steps, shift `5`, guide scale `5`
- elapsed: `17.202` seconds
- peak allocated VRAM: `12.182` GB
- peak reserved VRAM: `17.26` GB
- output: `renders/smoke_ti2v_5b_640x352_25f_2steps.mp4`

This proves the standalone runner can load the local WAN 5B fp16 diffusion model, UMT5 encoder, and WAN 2.2 VAE, then execute a complete segment without ComfyUI.

## Next Slice

The next backend slice is to harden the render path before enabling longer jobs:

- stream progress and cancellation state into the local API
- surface render lock state in the UI
- install or build a proper compiled attention kernel for the production runtime
- then implement A14B FP8 linear support and high/low expert loading
- apply profile LoRAs to the correct high/low expert only
- compare A14B 720p output duration, frame count, and VRAM against the 25 GB reference before enabling multi-segment rendering
