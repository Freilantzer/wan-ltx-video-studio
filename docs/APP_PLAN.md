# App Plan

## Product Concept

Create a focused local-first video generation studio for WAN 2.2, with a clean path to support LTX later. It should feel like a purpose-built creative tool, not a node graph and not a ComfyUI wrapper.

The app ships with its own standalone local render server. ComfyUI is technical inspiration only: a source of proven workflow anatomy, custom-node behavior, model placement clues, and performance lessons. It is not a runtime dependency, backend adapter, or required install.

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

This means the first-class target is not an 8 GB or 12 GB fallback profile. The app should default to high-performance WAN 2.2 workflows suitable for a 32 GB Blackwell card, while still offering lower-memory modes later. For 720p WAN 2.2 A14B I2V, the first direct renderer target is about 25 GB peak VRAM, based on the user's proven workflow.

## Design Principles

- Hide model-pipeline complexity without hiding creative power.
- Build a lean app for video generation, not a general workflow graph.
- Treat model files, LoRAs, encoders, VAEs, and upscalers as first-class library assets.
- Make memory/performance modes understandable and reversible.
- Always preserve generation metadata so good results can be reproduced.
- Keep runtime execution inside our app-managed direct renderer.
- Use ComfyUI only as a read-only reference when we need to understand a working technique.

## MVP Scope

### Generation Modes

- WAN 2.2 image-to-video using chunked segments.
- WAN 2.2 text-to-video or TI2V once the direct runtime path is validated.
- WAN 2.2 14B image-to-video presets for original and Lightning/turbo model profiles.
- WAN 2.2 5B fallback profile if useful for speed or memory.

### Core Controls

- Shared prompt or per-segment prompts.
- Negative prompt.
- Input image for I2V.
- Segment count, seconds per segment, FPS, and derived duration.
- Resolution and pixel-budget validation.
- Seed: random, fixed, increment per segment.
- Base model selector.
- LoRA stack with role labels: creative, workflow/turbo, control.
- Lightning/turbo profile that can treat performance LoRAs as part of the model profile.
- Motion continuity controls: motion frames and motion amplitude.

### Render Management

- Queue with one active GPU job by default.
- Cancel and retry.
- Per-segment progress events.
- Preview thumbnails or last completed segment.
- Output library with prompt, segment prompts, seed, model files, LoRAs, runtime version, and settings.
- Re-run, remix, reveal in folder, and compare.

## Recommended Architecture

```text
Desktop / Local Web UI
        |
        v
App Render Server
        |
        +-- Project DB / render history
        +-- Model library scanner
        +-- Preset registry
        +-- Job queue
        +-- Progress event stream
        +-- Media stitching / thumbnails
        +-- Direct renderer adapters
              |
              +-- Direct WAN renderer
              +-- Direct LTX renderer
```

The UI never talks to ComfyUI or raw model scripts. It talks to the app render server. The render server owns validation, job lifecycle, memory policy, direct inference calls, output collection, and metadata.

## Tech Stack Recommendation

For a Windows-first local app:

- UI: React + TypeScript + Vite.
- Desktop shell: Tauri after the browser-first app shape stabilizes.
- Backend/render server: Python, eventually FastAPI or another async server once queue/progress endpoints need it.
- DB: SQLite.
- Queue: local async worker with one active GPU job by default.
- Media handling: FFmpeg for concat, transcode, thumbnails, and later audio handling.
- Runtime environments: isolated project-managed Python environments for direct WAN and direct LTX.

Why this shape: Python fits the AI runtime ecosystem, React gives a polished control surface quickly, and a direct render server keeps the product lean instead of inheriting a general graph tool.

## Renderer Contract

Each model family should implement:

- `capabilities`: T2V, I2V, V2V, audio, controls, upscale, LoRA, turbo.
- `validateSettings(settings, hardware, installedModels)`.
- `prepareJob(plan)`.
- `renderSegment(segmentPlan, continuityState)`.
- `streamProgress(jobId)`.
- `collectOutputs(jobId)`.
- `releaseMemory(strategy)`.

Initial renderers:

- `DirectWanRenderer`: direct WAN 2.2 inference owned by this project.
- `DirectLtxRenderer`: direct LTX provider once WAN chunking is functional.

Shared normalized settings:

- prompt and per-segment prompts
- negative prompt
- seed policy
- width, height, frame count, fps
- input assets
- quality mode
- memory profile
- base model
- LoRA list
- output format

Model-specific settings stay in an advanced panel, for example WAN high/low phase settings or LTX-specific controls.

## Chunked WAN Strategy

The first production path should model the user's proven long-video workflow as typed app concepts:

- `SegmentPlan`: width, height, frame count, fps, seed, prompt, negative prompt, chunk index, and model phase settings.
- `ContinuityPlan`: start image, previous segment frames, motion frame count, motion amplitude, duplicate-boundary trim policy.
- `EnginePlan`: model files, text encoder, VAE, attention mode, LoRA stack, memory profile, and output format.

The render server should generate segment plans from segment count, segment duration, FPS, and resolution. It should render one segment at a time, feed previous output frames into the next segment, trim duplicate boundary frames, concatenate the final result, and save metadata.

ComfyUI's `PainterLongVideo`, SageAttention patching, and video helper behavior can inform the implementation, but the shipped runtime should be direct code owned by this project.

## Memory Profiles

### Conservative

- Prefer smaller WAN profiles.
- Lower frame counts and 480p/short 720p presets.
- Use FP8 text encoder where quality allows.
- Offload text encoder, VAE, or model blocks where needed.

### Balanced

- 720p or 1080p-class pixel budgets.
- Moderate frame counts.
- Direct renderer manages model residency based on queue state.

### Performance

- Default profile for the RTX 5090 target machine.
- For 32 GB VRAM and known-good model placement.
- 720p A14B I2V target: about 25 GB peak VRAM, with a warning above 28 GB.
- Prefer keeping reusable model components resident between segments.
- Avoid repeated unload/reload within one render job.
- Prioritize WAN 2.2 14B FP8/distilled workflows when installed.

### Turbo

- Requires compatible distilled models or Lightning/LightX2V LoRAs.
- Uses low-step presets.
- Can be represented as a model profile when performance LoRAs are effectively part of the workflow.
- Shows a visible compatibility note because distilled speed changes quality and prompt behavior.

### Experimental Max

- RTX 5090-specific experimental profile.
- Tests higher frame counts, higher resolutions, longer context, and model residency.
- Can use local wheel builds or source builds when prebuilt packages lag Blackwell support.
- Never becomes the default until measured stable.

## Model Library

The model library should detect:

- diffusion models / transformer weights
- VAE files
- text encoders
- clip vision models
- LoRAs
- checkpoints
- latent upscalers

It should support central model storage through configured project paths or symlinks. The app should not require users to arrange files in ComfyUI folders.

## LoRA Handling

WAN 2.2 needs two LoRA concepts:

- Creative LoRAs: style, character, motion, camera, look.
- Workflow/performance LoRAs: Lightning, LightX2V, distilled acceleration.

The UI should:

- Tag LoRAs by family: WAN, LTX, unknown.
- Tag LoRAs by role: creative, control, turbo/distill/workflow.
- Warn when a LoRA does not match the selected model family.
- For WAN 14B, allow phase-specific high-noise/low-noise LoRA assignment.
- Allow model profiles where workflow LoRAs are built in and not user-toggleable.
- Store LoRA strength per render in metadata.

## Future LTX Support

LTX should be a second provider, not a fork of the WAN UI.

Add capabilities:

- LTX image/video generation.
- LTX distilled profiles.
- LTX control LoRAs where supported.
- Spatial and temporal upscaler stages.
- LTX-safe dimension and frame-count validation.

The first LTX integration should be direct. Existing LTX Comfy nodes may be inspected for implementation ideas, but not used as the runtime backend.

## User Experience Layout

Primary screens:

- Generate: prompt, segment prompts, reference image, preset, LoRAs, quality, memory, queue.
- Library: models, LoRAs, missing dependencies, install checklist.
- Renders: output grid, metadata, compare, remix, reveal in folder.
- Settings: model paths, runtime paths, hardware profile, FFmpeg path, Git/project settings.

Generate screen structure:

- Left: prompt and shot settings.
- Center: preview/output and segment timeline.
- Right: model, LoRA, memory, and advanced controls.
- Bottom: queue/timeline/history strip.

## Implementation Phases

### Phase 0: Repo and Research

- Create project scaffold.
- Save research and app plan.
- Decide GitHub publishing details.

### Phase 1: App Shell And Planning

- Build Generate screen.
- Add segment planner.
- Add shared/per-segment prompt controls.
- Add base model and LoRA controls.
- Add planner API endpoint.

### Phase 2: Direct WAN Runtime Spike

- Keep runtime isolated from global Python.
- Validate Torch/CUDA/SageAttention/FlashAttention options on RTX 5090.
- Load the smallest practical WAN profile without rendering first.
- Render a minimal single-segment smoke test.
- Record VRAM, timing, package versions, model files, and output metadata.

### Phase 3: Render Server

- Add render job model.
- Add queue and cancellation.
- Add progress event stream.
- Render one WAN segment from a `SegmentPlan`.
- Store output and metadata.

### Phase 4: Chunked Rendering

- Render N segments in sequence.
- Pass previous output frames into the next segment.
- Trim duplicate boundaries.
- Concatenate final video.
- Store per-segment prompt, seed, model, LoRA, memory, and timing metadata.

### Phase 5: Model And LoRA Library

- Scan model directories.
- Show missing model checklist.
- Add LoRA tagging and compatibility warnings.
- Add model profile validation.

### Phase 6: Turbo And Optimization

- Add Lightning/LightX2V profile support.
- Add memory profiles.
- Add batch queue policies.
- Add model unload/free-memory controls.

### Phase 7: LTX Direct Renderer

- Add LTX model family.
- Add LTX-specific validation.
- Add LTX direct runtime spike.
- Add upscaler stages where useful.

## Open Decisions

- App form: browser localhost first, or Tauri desktop once core rendering works.
- Exact direct WAN runtime package layout.
- License.
- Whether first real renderer should start with 14B/turbo on RTX 5090, with 5B as fallback, or the other way around for faster iteration.
