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

The first executable runner path was `wan22_ti2v_5b_fp16`. It is a smoke-test path for the direct render loop because its diffusion safetensors match the upstream WAN module keys directly.

The A14B FP8 I2V path is now wired for the local scaled safetensors:

- `wan22_i2v_a14b_fp8_original`
- `wan22_i2v_a14b_fp8_lightning_workflow`

The runner loads the high-noise and low-noise experts as separate CPU-resident modules, patches FP8 scaled `Linear` layers with their `scale_weight` tensors, preserves WAN's timestep-based high/low expert switching, attaches phase-specific Lightning LoRAs when the Lightning profile is selected, and uses the WAN 2.1 VAE. The first GPU render is still pending; current validation is CPU-only model compatibility plus CLI dry-run.

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

## 5B Calibration Notes

The 2-step smoke output is only a wiring test and should not be used to judge visual quality. A five-render calibration pass at `640x352`, `25` frames, `24` fps showed:

| Run | Mode | Steps | Shift | Result |
| --- | --- | ---: | ---: | --- |
| `calib01` | start-frame I2V | 8 | 5 | recognizable and stable, mostly anchored to the source frame |
| `calib02` | start-frame I2V | 16 | 3 | clean and stable, slightly more natural at low resolution |
| `calib03` | text-only T2V | 16 | 5 | still under-formed and dark |
| `calib04` | text-only T2V | 40 | 5 | coherent alley composition, but noticeably stylized |
| `calib05` | start-frame I2V | 40 | 5 | best detail, but begins adding saturated light artifacts |

Working conclusion: the direct 5B fp16 path is useful for backend validation and small previews, especially with a start frame. It is not the quality target. The production-quality path should focus on A14B high/low expert loading, Lightning/Turbo LoRA routing, and matching the known 720p Comfy reference.

Meaningful WAN calibration runs should use `81` frames. Shorter runs are allowed as smoke tests, but the runner marks them with a warning in the JSON payload so they are not mistaken for quality measurements.

## Memory Telemetry

The direct runner emits staged CUDA memory snapshots under `telemetry.stages`. Each snapshot includes:

- current PyTorch allocated/reserved VRAM
- peak PyTorch allocated/reserved VRAM so far
- driver-level free/total/used VRAM from `torch.cuda.mem_get_info()` when available
- elapsed seconds since the runner entered GPU execution
- optional stage details such as component name or DiT forward call count

The current staged checkpoints cover CUDA setup, pipeline shell creation, text encoder load and forward calls, VAE load/encode/decode, DiT CPU load, DiT transfer to/from CUDA, selected DiT sampling forward calls, save, and cleanup. The top-level `peakAllocatedGb`, `peakReservedGb`, and `peakDriverUsedGb` fields remain as compact summary values.

The runner also writes the same staged snapshots to a sidecar file next to the target video, using the suffix `.telemetry.jsonl`. This is deliberate: if the process is killed or CUDA raises an out-of-memory error during VAE decode, the completed stage snapshots are still available for post-mortem.

On Windows, the runner also writes a `.windows-gpu-memory.jsonl` sidecar using `typeperf` GPU Adapter Memory counters. This tracks dedicated usage and shared GPU memory usage separately, which matters because WDDM can spill GPU allocations into shared system RAM instead of throwing an immediate CUDA OOM.

An 81-frame 1280x720 start-frame run with the 5B path completed all 8 sampling steps at about `24.1` GB dedicated VRAM, then VAE decode crossed the dedicated VRAM boundary and spilled into Windows shared GPU memory before the run was manually stopped. Conclusion: sampling fits, but full-frame VAE decode needs Comfy-style memory handling before 720p/81 can be considered supported.

The first standalone mitigation is a streaming temporal VAE decode path for WAN 2.2. Immediately before VAE decode, the runner explicitly pushes the DiT and text encoder to CPU, synchronizes, runs GC, and clears CUDA cache. Instead of concatenating each decoded temporal chunk into a growing CUDA tensor, the runner decodes one latent-time chunk at a time, moves that decoded chunk to CPU, clears CUDA cache pressure, and concatenates the final video on CPU. After decode, it moves the VAE itself to CPU before video saving.

Follow-up 81-frame 1280x720 test result: pre-VAE cleanup worked, dropping driver-used VRAM to about `3.5` GB before decode. Temporal streaming still failed at 720p because the VAE decoder's per-spatial-chunk activation cost crossed dedicated VRAM and spilled into shared GPU memory. The run reached about `31.1` GB dedicated plus `9.75` GB shared before it was stopped. Conclusion: temporal streaming is useful but insufficient; 720p/81 needs spatial tiled VAE decode or a Comfy-equivalent tiled VAE fallback.

The current runner adds that spatial tiled VAE path. It splits the WAN latent spatially, decodes each tile with the temporal streaming decoder, blends tile overlaps with feather masks, accumulates the final decoded video on CPU, and offloads the VAE before video saving.

Latest exact 720p/81 test result with the 5B path:

- output: `renders/tiled_720p_exact_i2v_alley_1280x720_81f_8steps.mp4`
- request: `1280x720`, `81` frames, `16` fps, `8` sample steps
- saved video: `1280x720`, `81` frames, `5.0625` seconds
- elapsed: `118.129` seconds
- peak runner driver-used VRAM: `21.488` GB
- peak Windows dedicated GPU memory: `21.925` GB
- peak Windows shared GPU memory: `0.154` GB
- VAE decode began and ended around `3.646` GB driver-used VRAM
- VAE tiling: `8` spatial tiles for the generated latent

A second exact-size 720p/81 start-frame run used `inputs/start_frames_1280x720/woman_black_sand_beach.png` and completed with the same tiled path:

- output: `renders/tiled_720p_exact_i2v_woman_1280x720_81f_8steps.mp4`
- saved video: `1280x720`, `81` frames, `5.0625` seconds
- elapsed: `116.25` seconds
- peak runner driver-used VRAM: `20.841` GB
- peak Windows dedicated GPU memory: `21.28` GB
- peak Windows shared GPU memory: `0.154` GB
- VAE decode began and ended around `3.527` GB driver-used VRAM

WAN TI2V 5B snaps 1280x720 to its internal spatial grid. The 5B runner currently generates at `1280x736`, resizes the start image to that grid, then center-crops the decoded result back to the requested `1280x720` before saving. This avoids the upstream floor-to-704 behavior while keeping the saved video at the requested size. The A14B I2V path uses WAN's native 720p grid for a 1280x720 start image and raises an error if the result does not match the requested size.

## A14B CPU Compatibility

The A14B Lightning path has passed a CPU-only load check against the local model files:

- FP8 UMT5 text encoder loads and patches scaled linears.
- WAN 2.1 VAE safetensors load into the upstream VAE class shape.
- Low-noise A14B expert loads with `406` FP8 parameters and `400` Lightning LoRA modules attached.
- High-noise A14B expert loads with `406` FP8 parameters and `400` Lightning LoRA modules attached.
- CLI dry-run resolves all A14B Lightning components without creating a CUDA context.

## Comfy Memory Findings

The reference Comfy behavior is not a single trick. It combines runtime-wide model residency management with VAE-specific tiled decode paths.

Core Comfy memory behavior observed in the read-only install:

- The launch script sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True` and starts Comfy with `--use-sage-attention --disable-smart-memory`.
- `--disable-smart-memory` does not disable offload. It makes Comfy's model manager aggressively offload instead of keeping models in VRAM when it can.
- `load_models_gpu()` budgets model memory plus inference memory, calls `free_memory()` before model loads, and can partially load/offload modules through low-VRAM patching.
- Standard `VAEDecode` calls `vae.decode(...)`; Comfy's VAE wrapper estimates decode memory, loads only the VAE patcher, and can retry with tiled VAE decode after a CUDA OOM.
- `VAEDecodeTiled` is an explicit node path, but the inspected workflow uses plain `VAEDecode`, so the working reference does not require the user to manually choose a tiled node.

Custom-node behavior in the reference install also matters:

- `comfyui-multigpu` patches `soft_empty_cache()` and some device-selection functions. On this one-GPU machine, the most relevant effect is broader cache cleanup and CPU/RAM pressure handling, not actual multi-GPU scheduling.
- `ComfyUI-WanVideoWrapper` is Apache-2.0 licensed and contains a WAN-specific tiled VAE decode implementation: split latent spatial tiles, decode one tile at a time, blend overlaps with feather masks, accumulate output on CPU, and offload the VAE afterward.

Implementation implication for this standalone renderer:

- Keep the current pre-VAE full offload step.
- Keep the spatial tiled WAN VAE decode path with overlap blending and CPU accumulation.
- Keep Windows dedicated/shared telemetry as the guardrail; success means avoiding shared GPU spill, not merely avoiding a Python OOM.
- Prefer an app-owned implementation of the concept. Copying Comfy core code would pull in GPLv3 obligations; WanVideoWrapper's relevant code is Apache-2.0, but a clean implementation is still easier to maintain.

## Next Slice

The next backend slice is to harden the render path before enabling longer jobs:

- stream progress and cancellation state into the local API
- surface render lock state in the UI
- surface staged VRAM telemetry in the UI/API render details
- run the first A14B Lightning GPU smoke and compare dedicated/shared VRAM against the 25 GB target
- tune tiled VAE presets for quality, speed, and A14B profiles
- install or build a proper compiled attention kernel for the production runtime
- compare A14B 720p output duration, frame count, and VRAM against the 25 GB reference before enabling multi-segment rendering
