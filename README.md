# WAN/LTX Video Studio

A planned local-first video generation app focused on WAN 2.2, with a clean path to support the current open LTX family later.

The goal is not to recreate a node graph UI. The app should expose a focused, powerful video-generation workspace while using proven workflow engines underneath, starting with ComfyUI in headless/API mode.

## Current Status

- Local project folder created at `D:\VIDEO_GENS\wan-ltx-video-studio`.
- Git repository initialized locally.
- Planning docs created under `docs/`.
- No GitHub remote has been created yet.

## Project Docs

- [App plan](docs/APP_PLAN.md)
- [Research notes](docs/RESEARCH_NOTES.md)
- [GitHub setup](docs/GITHUB_SETUP.md)

## Working Product Direction

Build a desktop or local web app with:

- WAN 2.2 text-to-video, image-to-video, and hybrid text/image-to-video presets.
- LoRA library management, including normal creative LoRAs and turbo/distillation LoRAs.
- Turbo mode built from validated distilled/4-step workflows, not magic settings.
- Hardware-aware memory modes for 8 GB, 12-16 GB, 24 GB, and high-VRAM machines.
- Queue, batch, prompt versioning, seed management, previews, and render history.
- A model/provider abstraction so LTX 2.x/2.3 can be added without rebuilding the UI.

## What I Need From You For GitHub

To publish this as a GitHub project, I need:

1. Repository name, unless `wan-ltx-video-studio` is okay.
2. Visibility: private or public.
3. License preference for our app code, or confirmation to leave it unlicensed for now.
4. GitHub auth path:
   - Install and authenticate GitHub CLI (`gh auth login`), or
   - Create an empty repo on GitHub and give me the remote URL.

`gh` is not currently installed on this machine, so I cannot create the GitHub repo from the terminal yet.

