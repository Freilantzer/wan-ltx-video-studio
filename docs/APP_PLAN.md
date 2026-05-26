# App Plan

## Product Concept

Create a focused local-first video generation studio for WAN 2.2. It should feel like a purpose-built creative tool, not a node graph. ComfyUI can provide an early execution engine, workflow compatibility, model discovery, progress events, and output retrieval; the app owns the product model, queue, presets, settings validation, render history, and model strategy.

Future LTX support should be designed in from the start through an engine/model adapter layer.

## Default Target Machine

The default performance target is the owner's local Windows machine:

- GPU: NVIDIA GeForce RTX 5090
- VRAM: 32 GB class, reported as 32607 MiB
- Driver: 591.86
- CUDA runtime reported by `nvidia-smi`: 13.1
- Active CUDA Toolkit in `CUDA_PATH`: 12.8
- Active `nvcc`: 12.8.61
- Python installs available: 3.10, 3.11, 3.12, 3.13
- Node: 24.11.1

This means the first-class target is not an 8 GB or 12 GB fallback profile. The app should default to high-performance WAN 2.2 workflows suitable for a 32 GB Blackwell card, while still offering lower-memory modes for portability.

## Design Principles

- Hide graph complexity without hiding power.
- Use validated workflow templates rather than arbitrary user-created graphs for the MVP.
- Treat model files, LoRAs, encoders, VAEs, and upscalers as first-class library assets.
- Make memory/performance modes understandable and reversible.
- Always preserve generation metadata so good results can be reproduced.
- Avoid becoming a thin ComfyUI wrapper. ComfyUI is an engine option, not the product.

## MVP Scope

### Generation Modes

- WAN 2.2 TI2V-5B text-to-video.
- WAN 2.2 TI2V-5B image-to-video.
- WAN 2.2 14B text-to-video preset if hardware/model validation passes.
- WAN 2.2 14B image-to-video preset if hardware/model validation passes.

### Core Controls

- Positive prompt and negative prompt.
- Input image for I2V.
- Resolution presets: 480p, 720p landscape, 720p portrait where workflow supports it.
- Duration/frame count with model-safe increments.
- FPS display and derived duration.
- Seed: random, fixed, reuse.
- Sampler/steps exposed through simple quality modes first.
- LoRA stack with strength sliders and compatibility warnings.
- Turbo toggle that switches to validated distilled workflow/model/LoRA combinations.

### Render Management

- Queue with pause/cancel/retry.
- Live progress through ComfyUI WebSocket events.
- Preview frames or thumbnails when available.
- Output library with prompt, seed, model files, LoRAs, workflow version, and settings.
- Compare two outputs side by side.
- Re-run, remix, or upscale from a past render.

## Recommended Architecture

```text
Desktop / Local Web UI
        |
        v
App API Server
        |
        +-- Project DB / render history
        +-- Model library scanner
        +-- Workflow preset registry
        +-- Engine adapters
              |
              +-- ComfyUI local adapter
              |     +-- POST /prompt
              |     +-- /ws progress
              |     +-- /history output retrieval
              |     +-- /system_stats and /models discovery
              |
              +-- Future direct WAN adapter
              +-- Future LTX adapter
```

## Tech Stack Recommendation

For a Windows-first local app:

- UI: React + TypeScript + Vite.
- Desktop shell: Tauri if we want a small native wrapper, or browser-first localhost for fastest iteration.
- Backend: Python FastAPI for workflow generation, ComfyUI control, file management, and hardware probing.
- DB: SQLite.
- Queue: local async worker with one active GPU job by default.
- Media handling: FFmpeg for transcoding, thumbnails, metadata, and waveform/audio inspection later.

Why this shape: Python already fits the AI tooling ecosystem, while React gives us a polished control surface quickly. Keeping ComfyUI as an optional managed sidecar avoids reimplementing rapidly moving video inference internals before we know which parts are worth owning directly.

## Engine Adapter Contract

Each model family should implement:

- `capabilities`: T2V, I2V, V2V, audio, controls, upscale, LoRA, turbo.
- `validateSettings(settings, hardware, installedModels)`.
- `buildWorkflow(settings)`.
- `submit(workflow)`.
- `streamProgress(jobId)`.
- `collectOutputs(jobId)`.
- `freeMemory(strategy)`.

Initial adapters:

- `ComfyWanAdapter`: submits generated API workflows to local ComfyUI.
- `DirectWanAdapter`: planned spike using WAN's native inference code once baseline Comfy execution works.
- `ComfyLtxAdapter`: planned LTX support through ComfyUI-LTXVideo and/or built-in Comfy LTX nodes.
- `DirectLtxAdapter`: optional later spike if direct LTX pipelines provide better control/performance.

Shared normalized settings:

- prompt, negative prompt
- seed
- width, height, frame count, fps
- input assets
- quality mode
- memory profile
- LoRA list
- output format

Model-specific settings stay in an advanced panel, for example WAN high/low phase settings or LTX audio/upscaler settings.

## Workflow Strategy

Keep workflows in `workflows/` as versioned API-format JSON templates, plus a typed manifest:

```json
{
  "id": "wan22-ti2v-5b-t2v",
  "engine": "comfyui",
  "family": "wan",
  "model": "Wan2.2-TI2V-5B",
  "mode": "t2v",
  "requires": {
    "diffusion_models": ["wan2.2_ti2v_5B_fp16.safetensors"],
    "vae": ["wan2.2_vae.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"]
  },
  "editableNodes": {
    "prompt": {"node": "CLIPTextEncode", "input": "text"},
    "seed": {"node": "KSampler", "input": "seed"},
    "frames": {"node": "Wan22ImageToVideoLatent", "input": "length"}
  }
}
```

The app should never edit a raw workflow by brittle string replacement. It should parse JSON, patch known node IDs or semantic bindings from the manifest, validate, then submit.

## Memory Profiles

### Conservative

- Prefer WAN 5B.
- Lower frame counts and 480p/short 720p presets.
- Use FP8 text encoder.
- Use ComfyUI auto/low VRAM behavior.
- Allow VAE/text encoder CPU/offload where needed.

### Balanced

- WAN 5B 720p or 14B FP8 when available.
- ComfyUI auto memory mode.
- Moderate frame counts.
- Default preview settings.

### Performance

- Default profile for the RTX 5090 target machine.
- For 32 GB VRAM and known-good model placement.
- Prefer keeping models resident.
- Avoid repeated unload/reload.
- Useful for batches with same model and LoRA stack.
- Prioritize WAN 2.2 14B FP8/distilled workflows when installed.

### Turbo

- Requires compatible LightX2V distilled models or LoRAs.
- Uses 4-step/low-step presets.
- Shows a visible compatibility note because distilled speed changes quality and prompt behavior.

### Experimental Max

- RTX 5090-specific experimental profile.
- Tests higher frame counts, higher resolutions, longer context, and model residency.
- Can use local wheel builds or source builds when prebuilt packages lag Blackwell support.
- Never becomes the default until measured stable.

## Model Library

The model library should detect:

- `diffusion_models`
- `vae`
- `text_encoders`
- `clip_vision`
- `loras`
- `checkpoints`
- `latent_upscale_models`

It should support external central model storage using ComfyUI's extra model paths rather than copying huge files.

## LoRA Handling

WAN 2.2 needs two LoRA concepts:

- Creative LoRAs: style, character, motion, camera, look.
- Performance LoRAs: LightX2V/Lightning/distilled LoRAs.

The UI should:

- Tag LoRAs by family: WAN, LTX, unknown.
- Tag LoRAs by role: creative, control, turbo/distill.
- Warn when a LoRA does not match the selected model family.
- For WAN 14B, allow phase-specific high-noise/low-noise LoRA assignment.
- Store LoRA strength per render in metadata.

## Future LTX Support

LTX 2.3 should be a second provider, not a fork of the WAN UI.

Add capabilities:

- Audio-video generation.
- LTX distilled mode, 8 steps, CFG=1.
- LTX IC-LoRA controls: depth, canny/edge, pose, motion tracking, HDR/lipdub where supported.
- Spatial and temporal upscaler stages.
- LTX-safe dimension and frame-count validation.

The first LTX integration should use ComfyUI-LTXVideo workflows, then later direct LTX pipelines if that becomes clearly better.

## User Experience Layout

Primary screens:

- Generate: prompt, reference image, preset, LoRAs, quality, memory, queue.
- Library: models, LoRAs, missing dependencies, download/install checklist.
- Renders: output grid, metadata, compare, remix, reveal in folder.
- Settings: ComfyUI path, model paths, hardware profile, FFmpeg path, Git/project settings.

Generate screen structure:

- Left: prompt and shot settings.
- Center: preview/output.
- Right: model, LoRA, memory, and advanced controls.
- Bottom: queue/timeline/history strip.

## Implementation Phases

### Phase 0: Repo and Research

- Create project scaffold.
- Save research and app plan.
- Decide GitHub publishing details.

### Phase 1: ComfyUI Control Spike

- Detect/running ComfyUI.
- Read `/system_stats`, `/models`, and `/object_info`.
- Submit one known WAN 5B workflow.
- Stream progress and collect output.
- Record exact VRAM usage and timing on RTX 5090.

### Phase 1B: Chunked Video Pipeline

- Translate the working WAN 2.2 long-video workflow into app-level concepts.
- Generate segment plans from total duration, chunk duration, FPS, and resolution.
- Render each segment with previous-frame continuity.
- Trim duplicate boundary frames between segments.
- Concatenate segments automatically.
- Store per-segment metadata, seed, model files, memory profile, and prompt.
- Keep the raw Comfy graph hidden behind typed settings.

### Phase 1.5: Direct WAN Spike

- Install WAN 2.2 native inference environment in an isolated project environment.
- Verify PyTorch CUDA support on RTX 5090.
- Run the smallest practical WAN 2.2 command-line generation.
- Compare direct WAN vs ComfyUI on setup complexity, speed, memory behavior, LoRA handling, and output management.
- Decide whether MVP execution should be Comfy-first, direct-first, or dual-backend.

### Phase 2: MVP UI

- Build Generate screen.
- Add workflow preset manifest.
- Add settings validation.
- Add render history DB.

### Phase 3: Model and LoRA Library

- Scan model directories.
- Show missing model checklist.
- Add LoRA tagging and compatibility warnings.
- Add external model path guidance.

### Phase 4: Turbo and Optimization

- Add LightX2V distilled workflow support.
- Add memory profiles.
- Add batch queue policies.
- Add model unload/free-memory controls.

### Phase 5: LTX Adapter

- Add LTX 2.3 model family.
- Add LTX ComfyUI workflow presets.
- Add audio output handling.
- Add spatial/temporal upscaler workflows.

## Open Decisions

- App form: browser localhost first, or Tauri desktop from the beginning.
- Whether the app should install/manage ComfyUI or only connect to an existing install.
- GitHub visibility and license.
- Whether first MVP should prioritize 14B/turbo on RTX 5090, with 5B as fallback.
- Whether ComfyUI remains the primary runtime or becomes an import/export compatibility layer after the Direct WAN spike.
