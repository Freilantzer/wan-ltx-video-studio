# WAN/LTX Video Studio

WAN/LTX Video Studio is a local-first video generation app for focused WAN 2.2 production, with a path to add open LTX models later.

The goal is a lean studio for prompt-driven and image-driven video work: segment planning, model selection, LoRAs, turbo profiles, VRAM-aware rendering, output history, and repeatable long-video workflows. The app owns its local render server and uses proven ComfyUI workflows as technical reference material.

## Project State

The project lives at:

```text
D:\VIDEO_GENS\wan-ltx-video-studio
```

The GitHub repository is:

```text
git@github.com:Freilantzer/wan-ltx-video-studio.git
```

Current working pieces:

- React/Vite local web app under `apps/web/`.
- Python package under `src/wan_ltx_studio/`.
- Local development API with render planning endpoints.
- Segment planner for 1 to many chunked video segments.
- Model/profile metadata for WAN 2.2 and future LTX providers.
- Local model-library mapping with large model files ignored by git.
- Direct WAN 2.2 5B runner for smoke tests and calibration.
- GPU opt-in guard for render execution.
- Staged CUDA memory telemetry written into render results.
- Windows dedicated/shared GPU memory sidecar telemetry via `typeperf`.
- Documentation for architecture, VRAM targets, local models, reference workflow analysis, and direct renderer behavior.

## Current Renderer

The first executable backend path is `wan22_ti2v_5b_fp16`. It proves the app-owned renderer can load local WAN model files, encode prompts/start frames, run sampling, decode VAE output, and save MP4s directly.

Completed calibration work:

- 640x352 smoke render completed successfully.
- 5B start-frame and text-only calibration renders completed.
- 81-frame 1280x720 sampling reached the end of diffusion at about 24 GB dedicated VRAM.
- Pre-VAE offload now moves the DiT and text encoder to CPU before decode.
- Temporal VAE streaming was implemented and measured.
- Spatial tiled WAN VAE decode with overlap blending and CPU accumulation was implemented.
- Two 81-frame 1280x720 I2V calibration renders now complete with exact 1280x720 saved frames.

Latest 720p finding:

The direct 5B runner completed exact 81-frame 1280x720 I2V calibration renders with tiled VAE decode on both environment and portrait start frames. The runs generate on the WAN spatial grid, center-crop back to 1280x720, and stayed around 21.3 to 21.9 GB Windows dedicated GPU memory with about 0.15 GB shared GPU memory. Shared GPU memory remains a hard warning signal for future A14B work.

## Target Workflow

The first serious production target is based on the user's proven WAN 2.2 I2V workflow:

- WAN 2.2 A14B high-noise and low-noise experts.
- 1280x720.
- 81 frames per segment.
- 16 fps.
- 1 to 5 practical segments, with longer chains possible through continuity.
- Target dedicated VRAM around 25 GB on the RTX 5090 32 GB machine.
- Per-segment prompts, seeds, model choices, LoRAs, timing, and VRAM telemetry.
- Last-frame or continuity-frame carryover between segments.
- Duplicate boundary-frame trimming before final concat.

## Open Gaps

The main items still ahead:

- API progress stream and cancellation state.
- Render details in the UI, including telemetry and output metadata.
- A14B FP8 scaled weight support.
- High/low expert loading with only the active expert on CUDA.
- Lightning/Turbo LoRA routing for the correct expert.
- User-facing model and LoRA selectors wired to the render plan.
- LTX provider implementation.
- Packaged local desktop experience.

## Development

Run the Python tests:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests
```

Run the development API:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\scripts\run_dev_api.py --host 127.0.0.1 --port 8787
```

Run the web app:

```powershell
cd .\apps\web
npm install
npm run dev -- --port 5173
```

Open:

```text
http://127.0.0.1:5173/
```

Run a safe renderer dry-run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m wan_ltx_studio.rendering.single_segment_runner --dry-run --profile wan22_ti2v_5b_fp16 --prompt "test" --output renders\dry_run.mp4
```

Real GPU renders require `--allow-gpu`.

## Project Docs

- [App plan](docs/APP_PLAN.md)
- [Architecture decisions](docs/ARCHITECTURE_DECISIONS.md)
- [Direct renderer backend](docs/DIRECT_RENDERER_BACKEND.md)
- [Chunked video planner](docs/CHUNKED_VIDEO_PLANNER.md)
- [VRAM targets](docs/VRAM_TARGETS.md)
- [Research notes](docs/RESEARCH_NOTES.md)
- [Reference workflow analysis](docs/REFERENCE_WORKFLOW_ANALYSIS.md)
- [Read-only Comfy reference install](docs/READ_ONLY_COMFY_REFERENCE_INSTALL.md)
