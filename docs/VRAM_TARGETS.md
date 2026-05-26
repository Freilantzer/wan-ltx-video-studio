# VRAM Targets

Date: 2026-05-26

The first direct renderer target should be based on the user's proven 720p WAN 2.2 workflow, not on the earlier pressure test.

## 720p A14B I2V Target

Target profile:

- Model family: WAN 2.2 I2V A14B high/low experts.
- Resolution: 1280 x 720.
- Segment size: 81 input frames at 16 fps.
- Timeline shape: 1 to 5 segments, with 3 x 5 seconds already proven.
- Target dedicated VRAM: about 25 GB.
- Warning threshold: above 28 GB.
- Unsafe threshold: above 30 GB unless explicitly running an experimental profile.

The user observed a working 3 x 5 second output at 1280 x 720 with about 25.1 GB dedicated VRAM in Task Manager.

## Reference Ingredients

Read-only start script:

```text
D:\IMAGE_GENERATORS\Comfy_UI_Furkan_V61\Windows_Run_GPU_optimized.bat
```

Active runtime settings in that script:

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
--windows-standalone-build --use-sage-attention --disable-smart-memory
```

Relevant workflow traits:

- FP8 text encoder.
- WAN 2.1 VAE.
- Two A14B FP8 high/low model experts.
- The selected model files are custom lighting variants, so the app should treat "lighting included in base model" as a valid model profile.
- Lightning/creative LoRA loader nodes are present in the workflow, but the inspected LoRA nodes are bypassed.
- SageAttention is patched per model in the workflow and also enabled at startup.
- Sampling is 4 total steps split into high-noise and low-noise phases.
- Later segments reuse previous decoded frames through the continuity stage.
- Duplicate first frames are trimmed from later segments before final concat.

## Direct Renderer Requirements

The standalone renderer should reproduce the memory behavior with app-owned code:

- Use chunked segment rendering rather than one monolithic long latent.
- Load high/low experts as model-profile components.
- Support model profiles where lighting or turbo behavior is built into the selected base model.
- Support optional LoRAs separately from built-in model-profile behavior.
- Use SageAttention or equivalent optimized attention when available.
- Use PyTorch allocator settings equivalent to `max_split_size_mb:256,expandable_segments:True` where appropriate.
- Keep VAE/text encoder/model residency explicit and measurable.
- Offload text encoder and diffusion experts before VAE decode.
- Decode 720p/81 VAE output with spatial tiling, overlap blending, and CPU accumulation.
- Record peak VRAM per segment and per full job.
- Treat Windows shared GPU memory usage as an unsafe spill condition, even if CUDA does not throw an OOM.

The acceptance target for the first serious 720p direct A14B I2V renderer is not "fits in 32 GB"; it is "stays around the proven 25 GB profile."
