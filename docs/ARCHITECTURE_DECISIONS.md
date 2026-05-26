# Architecture Decisions

## ADR-001: Standalone Product, Not A ComfyUI Wrapper

Status: accepted

The app must not be a simplified skin over ComfyUI, and it must not rely on ComfyUI as its rendering backend. It should own the product experience and the rendering server:

- generation modes
- model and LoRA library
- preset validation
- render queue
- output history
- metadata and reproducibility
- memory/performance profiles
- future WAN/LTX provider abstraction

ComfyUI remains useful as read-only technical inspiration. Its workflows and custom nodes can teach us how proven video pipelines handle conditioning, chunking, LoRAs, attention patches, VAE memory pressure, and output stitching. Those ideas should be translated into app-owned direct renderer code, not delegated to ComfyUI at runtime.

## ADR-002: RTX 5090 Is The Default Performance Target

Status: accepted

The owner's machine is the default target:

- NVIDIA GeForce RTX 5090
- 32607 MiB VRAM
- NVIDIA driver 591.86
- CUDA runtime reported by `nvidia-smi`: 13.1
- active CUDA Toolkit: 12.8
- active `nvcc`: 12.8.61

The app should therefore optimize first for 32 GB Blackwell workflows. Low-VRAM support remains important, but it should not define the main experience.

## ADR-003: Direct Renderer Is The Runtime Path

Status: accepted

The rendering server should execute WAN/LTX directly inside project-managed runtimes. ComfyUI is not a backend option for the product.

The first runtime milestone should focus on direct WAN execution and measure:

- install complexity
- PyTorch/CUDA reliability
- VRAM usage
- generation speed
- LoRA support
- turbo/distilled support
- error handling
- ability to support LTX later

ComfyUI experiments already performed are historical reference only. They are useful for understanding working settings and memory behavior, but they do not define the app runtime architecture.

## ADR-004: Use Isolated Runtimes

Status: accepted

Do not pollute the global Python installs. Create isolated environments for:

- app backend
- direct WAN runtime
- direct LTX runtime
- optional tooling/probe runtimes

This matters because Windows, Blackwell, CUDA, PyTorch, Triton, xFormers, SageAttention, flash-attention alternatives, and custom nodes can have conflicting dependency requirements.

## ADR-005: Build Long Video As A Segment Pipeline

Status: accepted

Long videos should be generated as chunked segment jobs, not as a single monolithic graph by default. The user's working WAN 2.2 workflow proves that 3 x 5 second chunks at 1280 x 720 can run on the target RTX 5090 while staying around the 25 GB VRAM range observed in Task Manager.

The app should own:

- segment planning from total duration, FPS, frame count, and pixel budget
- previous-frame or previous-video continuity
- duplicate boundary trimming
- final concatenation
- per-segment timing, seed, model, LoRA, and memory metadata

ComfyUI nodes such as `PainterLongVideo`, `PathchSageAttentionKJ`, and `VHS_VideoCombine` are useful references for how chunking, attention, and video output can work. The shipped app should implement equivalent concepts in its own planner, direct renderer, media pipeline, and metadata system.

## ADR-006: Reference Comfy Install Is Read-Only Inspiration

Status: accepted

The existing install at `D:\IMAGE_GENERATORS\Comfy_UI_Furkan_V61\ComfyUI` is a reference environment only. It should not be modified by this project.

The project may inspect it to understand node behavior, model placement, and proven workflow patterns. Runtime experiments, dependency installs, generated outputs, and managed app execution should happen inside `D:\VIDEO_GENS\wan-ltx-video-studio` or other explicitly created project directories.

## ADR-007: App Server Becomes The Render Server

Status: accepted

The local Python API should grow into the rendering server. The UI should submit render jobs to this server, and the server should own:

- queueing and cancellation
- one active GPU render by default
- model and LoRA resolution
- segment-by-segment direct inference
- progress events
- output stitching through FFmpeg or direct media utilities
- render metadata and history
- memory cleanup and model residency policy

This keeps the product lean and purpose-built while avoiding the complexity and visual-node mental model of a general workflow tool.
