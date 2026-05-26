# Research Notes

Research date: 2026-05-26

## Primary Sources

- WAN 2.2 GitHub: https://github.com/Wan-Video/Wan2.2
- WAN 2.2 TI2V-5B model card: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- ComfyUI WAN 2.2 workflow guide: https://docs.comfy.org/tutorials/video/wan/wan2_2
- ComfyUI WAN 2.2 examples: https://comfyanonymous.github.io/ComfyUI_examples/wan22/
- ComfyUI workflow concept docs: https://docs.comfy.org/development/core-concepts/workflow
- ComfyUI server routes: https://docs.comfy.org/development/comfyui-server/comms_routes
- ComfyUI server config: https://docs.comfy.org/interface/settings/server-config
- LightX2V WAN 2.2 distilled models: https://huggingface.co/lightx2v/Wan2.2-Distill-Models
- LightX2V WAN 2.2 distilled LoRAs: https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- LTX-2 GitHub: https://github.com/Lightricks/LTX-2
- ComfyUI-LTXVideo: https://github.com/Lightricks/ComfyUI-LTXVideo

## WAN 2.2 Findings

WAN 2.2 has several relevant variants:

- `Wan2.2-TI2V-5B`: hybrid text/image-to-video, 720p, 24 fps, high-compression VAE, Apache 2.0.
- `Wan2.2-T2V-A14B`: text-to-video MoE model, supports 480p and 720p.
- `Wan2.2-I2V-A14B`: image-to-video MoE model, supports 480p and 720p.
- `Wan2.2-S2V-14B`: speech-to-video.
- `Wan2.2-Animate-14B`: character animation and replacement.

The A14B line uses a high-noise/low-noise expert split. The practical consequence for our app: a "WAN 14B" preset is not a single checkpoint picker. It needs to manage paired model files, paired LoRAs, and phase-specific sampler settings.

The 5B TI2V model is the best first MVP target because it covers T2V and I2V in one family and is the most realistic consumer-GPU starting point.

## ComfyUI Workflow Findings

ComfyUI workflows are JSON graph files. Nodes load models, encode prompts, build latents, sample, decode, and save media. ComfyUI can also run as a local API endpoint:

- `POST /prompt` queues a prompt/workflow.
- `GET /history/{prompt_id}` retrieves outputs.
- `GET /system_stats` reads device/VRAM data.
- `GET /models/{folder}` lists installed model files.
- `POST /interrupt` cancels execution.
- `POST /free` unloads models/free memory.
- `/ws` streams progress events.

For this app, we should keep the node graph hidden and generate API-format workflow JSON from typed presets. Users edit "shot settings"; the app edits node inputs.

## WAN 2.2 ComfyUI Model Placement

From ComfyUI's WAN guide:

- 5B diffusion model: `ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`
- 5B VAE: `ComfyUI/models/vae/wan2.2_vae.safetensors`
- Text encoder: `ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`

For 14B T2V:

- High-noise diffusion model in `diffusion_models`
- Low-noise diffusion model in `diffusion_models`
- Text encoder in `text_encoders`
- VAE in `vae`

The app should support external model libraries through `extra_model_paths.yaml`, because model files are huge and should not live inside this repo.

## Turbo / Distillation Findings

LightX2V provides WAN 2.2 distilled models and LoRAs:

- Distilled 4-step models.
- FP8 and INT8 variants around 15 GB per high/low component.
- BF16 variants around 28.6 GB per component.
- ComfyUI-ready FP8 variants.
- Distilled LoRAs that can be loaded online or merged offline.

App implication:

- "Turbo" should be an explicit preset family with compatible model/LoRA pairs.
- The UI should show when turbo changes prompt adherence or quality expectations.
- The engine should validate high/low LoRA pairing for A14B workflows.
- Prefer pre-merged/quantized models for speed presets where available.

## Memory and Optimization Findings

ComfyUI exposes memory strategy controls such as `auto`, `lowvram`, `normalvram`, `highvram`, `novram`, and `cpu`, plus precision settings for UNET, VAE, and text encoder.

WAN's own inference flags include `--offload_model`, `--convert_model_dtype`, and `--t5_cpu`; ComfyUI's native offloading and model management cover similar concerns for the Comfy backend.

Initial app memory profiles:

- `Conservative`: lower resolution/frame count, CPU/offload-friendly, VAE/text encoder savings.
- `Balanced`: 5B or FP8 14B workflows, default Comfy smart memory.
- `Performance`: high-VRAM mode, keep models resident, fewer unloads.
- `Turbo`: distilled/4-step models or LoRAs, reduced steps, FP8/INT8 where validated.

## LTX Compatibility Findings

LTX-2.3 is the current open LTX line found during research. It is a 22B audio-video model with:

- Full dev model.
- Distilled model, 8 steps, CFG=1.
- Distilled LoRA.
- Spatial and temporal latent upscalers.
- Text/image/video/audio modes depending on workflow.
- ComfyUI support through built-in nodes plus `ComfyUI-LTXVideo` custom nodes for advanced workflows.

LTX uses a different capability surface from WAN:

- It can generate synchronized audio/video.
- It uses LTX-specific LoRAs and IC-LoRAs for control.
- It has spatial/temporal upscaler stages.
- LTX-2.3 dimensions must be divisible by 32; frame counts must follow the 8n+1 pattern.

App implication: do not bake WAN assumptions into the UI model. Use a provider adapter with shared fields where possible and model-specific settings where necessary.

