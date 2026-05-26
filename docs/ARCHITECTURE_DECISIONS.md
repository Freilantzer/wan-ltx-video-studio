# Architecture Decisions

## ADR-001: Product-Led App, Not a ComfyUI Wrapper

Status: accepted

The app must not be a simplified skin over ComfyUI. It should own the product experience:

- generation modes
- model and LoRA library
- preset validation
- render queue
- output history
- metadata and reproducibility
- memory/performance profiles
- future WAN/LTX provider abstraction

ComfyUI is useful because it already has fast-moving video model support, workflow execution, progress events, custom nodes, model discovery, and a local API. That makes it a strong first execution backend.

But ComfyUI is not the product boundary. The app should call into an engine adapter, and ComfyUI should be only one implementation of that adapter.

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

## ADR-003: Compare ComfyUI and Direct Inference Early

Status: accepted

The first technical milestone should not blindly commit to ComfyUI. It should compare:

- ComfyUI WAN execution through API workflows.
- Direct WAN 2.2 inference through the model repository code.

The decision should be based on measured behavior on the RTX 5090:

- install complexity
- PyTorch/CUDA reliability
- VRAM usage
- generation speed
- LoRA support
- turbo/distilled support
- error handling
- ability to support LTX later

Expected starting bias: ComfyUI is likely faster to integrate and easier to keep current, while direct inference may provide better ownership and a cleaner non-node product model if the upstream code is stable enough.

## ADR-004: Use Isolated Runtimes

Status: accepted

Do not pollute the global Python installs. Create isolated environments for:

- app backend
- ComfyUI runtime, if managed by this project
- direct WAN runtime
- direct LTX runtime, later

This matters because Windows, Blackwell, CUDA, PyTorch, Triton, xFormers, SageAttention, flash-attention alternatives, and custom nodes can have conflicting dependency requirements.

