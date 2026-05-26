# Direct Renderer Backend

The backend target is a standalone render server, not a ComfyUI wrapper. ComfyUI is only a reference for proven WAN 2.2 behavior on the target RTX 5090.

## Reference Contract

The known-good reference run is WAN 2.2 I2V A14B at 1280x720, 16 fps, three 81-frame input chunks, about 15.06 seconds of output, and roughly 25 GB dedicated VRAM. The backend treats that as the memory target for the A14B 720p profile:

- target: 25 GB
- warning: 28 GB
- unsafe: 30 GB

The direct renderer must keep the high-noise and low-noise A14B experts mutually exclusive on CUDA. The text encoder can be moved to CUDA for encoding and then returned to system RAM before diffusion, and VAE decode should happen after DiT experts are offloaded.

## Current Backend Slice

The first backend slice is deliberately GPU-safe. It resolves local model files, builds render-job commands, and exposes memory policy without loading models.

Endpoints:

- `GET /api/render/profiles`
- `POST /api/render/plan`

The render plan includes:

- selected WAN/LTX profile
- required model components and resolved paths
- built-in workflow LoRAs, such as Lightning profile components
- requested creative LoRAs
- segment commands with frame count, prompt, seed, trim, sample steps, and offload flags
- VRAM policy and allocator settings
- execution stages, with GPU execution marked pending

## Implementation Notes

The upstream WAN runtime already has a useful reference behavior in `WanI2V._prepare_model_for_timestep`: it chooses the active high/low expert by timestep and moves the inactive expert to CPU when offloading is enabled. The production backend should preserve that behavior, then adapt loading to our local Comfy-style safetensors and GGUF assets.

The current planned allocator matches the working reference script:

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
```

## Next Slice

The next backend slice is the actual direct WAN runner:

- add a process-level GPU job lock so only one render can load models at a time
- load Comfy-style WAN safetensors into the upstream WAN module shape
- apply profile LoRAs to the correct high/low expert only
- run one segment at 720p with telemetry around peak allocated/reserved VRAM
- compare output duration, frame count, and VRAM against the 25 GB reference before enabling multi-segment rendering
