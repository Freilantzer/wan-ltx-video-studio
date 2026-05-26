# WAN/LTX Video Studio

A planned local-first video generation app focused on WAN 2.2, with a clean path to support the current open LTX family later.

The goal is not to recreate a node graph UI. The app should expose a focused, powerful video-generation workspace with its own standalone local render server. ComfyUI is reference material for workflow behavior and node implementation details, not a runtime dependency or backend target.

## Current Status

- Local project folder created at `D:\VIDEO_GENS\wan-ltx-video-studio`.
- Git repository initialized and pushed to GitHub.
- Planning docs and architecture decisions created under `docs/`.
- Local model library mapped under `models/` and ignored by git.
- First app-owned planning code added under `src/`.
- First runnable local app scaffold added under `apps/web/`.

## Project Docs

- [App plan](docs/APP_PLAN.md)
- [Architecture decisions](docs/ARCHITECTURE_DECISIONS.md)
- [Chunked video planner](docs/CHUNKED_VIDEO_PLANNER.md)
- [VRAM targets](docs/VRAM_TARGETS.md)
- [Research notes](docs/RESEARCH_NOTES.md)
- [Reference workflow analysis](docs/REFERENCE_WORKFLOW_ANALYSIS.md)
- [Read-only Comfy reference install](docs/READ_ONLY_COMFY_REFERENCE_INSTALL.md)

## Working Product Direction

Build a desktop or local web app with:

- WAN 2.2 text-to-video, image-to-video, and hybrid text/image-to-video presets.
- LoRA library management, including normal creative LoRAs and turbo/distillation LoRAs.
- Turbo mode built from validated distilled/4-step workflows, not magic settings.
- Hardware-aware memory modes for 8 GB, 12-16 GB, 24 GB, and high-VRAM machines.
- Queue, batch, prompt versioning, seed management, previews, and render history.
- A model/provider abstraction so LTX 2.x/2.3 can be added without rebuilding the UI.

## Development

Run the Python tests with:

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

Then open:

```text
http://127.0.0.1:5173/
```

The current app does not start ComfyUI or load models. It plans segment timelines through the local API.
