# Runtime Bake-Off

Started: 2026-05-26

Goal: compare WAN 2.2 direct inference with WAN 2.2 through ComfyUI on the RTX 5090 target machine before deciding which runtime should power the MVP.

## Isolation Strategy

Runtime folders are local-only and ignored by Git:

```text
runtimes/
  direct-wan/
    src/Wan2.2/
    .venv/
  comfyui/
    src/ComfyUI/
    .venv/
  comfyui-sec-v89/
    src/ComfyUI/
    .venv/
  secourses-comfyui-v89-inspect/
  shared-models/
```

The app repository should track scripts, manifests, docs, and later source code. It should not track cloned runtime repos, model weights, virtual environments, or generated media.

## Test Questions

- Does PyTorch with CUDA 12.8+ detect RTX 5090 correctly?
- Can direct WAN 2.2 install cleanly on Windows?
- Can ComfyUI install cleanly on Windows with the same GPU stack?
- Which path handles WAN 2.2 14B, FP8/distilled, LoRA, and turbo workflows with less friction?
- Which path gives better telemetry and memory control?
- Which path is easier to hide behind a purpose-built app UX?

## Results Log

### Environment Baseline

See `docs/ENVIRONMENT_BASELINE.md`.

### Direct WAN

Status: smoke test passed, no model generation yet.

Runtime:

```text
runtimes/direct-wan
Python 3.10.11
Torch 2.12.0+cu130
CUDA built: 13.0
GPU detected: NVIDIA GeForce RTX 5090
```

Notes:

- Installed WAN 2.2 editable from a shallow upstream clone.
- Installed core WAN dependencies while avoiding `flash_attn` source build.
- Added missing practical import dependencies not covered by base requirements: `einops`, `decord`, `librosa`, `peft`, `sentencepiece`.
- `scripts/probe_wan_runtime.py` passes and sees all WAN 2.2 task configs.
- `generate.py --help` works.
- `flash_attn` is not installed in this runtime; WAN falls back to PyTorch scaled-dot-product attention.
- No model weights have been downloaded yet.

Risk:

- Upstream direct WAN imports optional S2V/Animate modules at top level, so extra optional dependencies are needed even for T2V/TI2V smoke tests.
- Direct WAN does not currently have packaged Windows FlashAttention for Torch 2.12/cu130 through normal pip.

### ComfyUI

Status: two ComfyUI runtime candidates tested.

#### Clean ComfyUI Runtime

Runtime:

```text
runtimes/comfyui
Python 3.10.11
Torch 2.12.0+cu130
CUDA built: 13.0
ComfyUI: 0.22.0
GPU detected: NVIDIA GeForce RTX 5090
```

Result:

- ComfyUI starts on localhost.
- `/system_stats` and `/object_info` work.
- Comfy sees 32607 MB VRAM and 738 node types.
- Comfy's CUDA backend is enabled after moving from cu128 to cu130.
- Triton is not available in this runtime.
- Attention defaults to PyTorch attention.

This is the cleanest official-style runtime and should remain the stability baseline.

#### SEcourses V89-Style Optimized Runtime

Source archive:

```text
C:\Users\freil\Downloads\ComfyUI_V89.zip
SHA256: 9595BB250068CC9ED8A306A16D7926D6C3349DEEFBCE9BEB806CEA8C6FC69CDC
```

Inspection notes:

- The archive contains scripts, requirements, workflows, and presets; it does not embed the acceleration wheels directly.
- `Windows_Install_Or_Update_ComfyUI.bat` installs Torch 2.9.1 from the cu130 PyTorch index.
- It then downloads third-party Windows wheels from `https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/`.
- Relevant wheels:
  - `flash_attn-2.8.3+torch2.9.1.cuda13.1-cp310-cp310-win_amd64.whl`
  - `xformers-0.0.34+41531cee.d20260109-cp39-abi3-win_amd64.whl`
  - `sageattention-2.2.0+torch2.9.1.cuda13.1-cp39-abi3-win_amd64.whl`
- `requirements_Comfy.txt` installs `triton-windows<3.6` on Windows.
- `Clear_Triton_Cache.py` deletes `~/.triton` and Windows temp contents; do not run it automatically.

Runtime:

```text
runtimes/comfyui-sec-v89
Python 3.10.11
Torch 2.9.1+cu130
CUDA built: 13.0
triton-windows: 3.5.1
flash-attn: 2.8.3
sageattention: 2.2.0
xformers: 0.0.34+41531cee.d20260109
ComfyUI: 0.22.0
```

Result:

- `pip check` reports no broken requirements.
- `scripts/probe_acceleration_packages.py` imports `triton`, `flash_attn`, `sageattention`, and `xformers`.
- ComfyUI starts on localhost with `--highvram --use-sage-attention`.
- `/system_stats` and `/object_info` work.
- Comfy logs confirm `Using sage attention`.
- Comfy's CUDA backend and Triton package are present; Comfy reports the Triton backend available but disabled for comfy-kitchen operations, while CUDA and eager backends are enabled.

This is the best current optimized Comfy runtime candidate for the RTX 5090, but it relies on third-party prebuilt wheels and an older Torch than the clean runtime.

## Workflow Preset Findings From V89 Archive

The V89 archive includes useful WAN 2.2 and LTX 2.3 preset workflows.

WAN 2.2 presets:

- `Wan 22 High Quality I2V 20 Steps.json`
- `Wan 22 High Quality T2V 20 Steps.json`
- `Wan 22 Image Realism.json`
- `Wan 22 Image To Video 4 Steps HQ.json`
- `Wan 22 Text To Video 8 Steps.json`

Common WAN nodes include:

- `UNETLoader`
- `CLIPLoader`
- `VAELoader`
- `CLIPTextEncode`
- `ModelSamplingSD3`
- `WanImageToVideo`
- `EmptyHunyuanLatentVideo`
- `SwarmKSampler`
- `CreateVideo`
- `SaveVideo`
- `RIFE VFI`

LTX 2.3 presets:

- `LTX2.3 Text To Video 8 Steps - 260426.json`
- `LTX2.3 Image To Video 8 Steps - 260426.json`
- `LTX2.3 Video To Video 8 Steps - 260426.json`
- audio/lip-sync and IC-LoRA variants

Common LTX nodes include:

- `LTXAVTextEncoderLoader`
- `LTXVConditioning`
- `LTXVConcatAVLatent`
- `LTXVSeparateAVLatent`
- `LTXVAudioVAELoader`
- `LTXVAudioVAEDecode`
- `LTXVImgToVideoInplace`
- `LTXVLatentUpsampler`
- `SwarmKSampler`
- `CreateVideo`
- `SaveVideo`
- video/audio helper nodes for advanced workflows

App implication:

- Treat these presets as references, not product architecture.
- For WAN MVP, prefer a lean workflow using native/official Comfy nodes where possible.
- If using V89 presets directly, we need SwarmUI ExtraNodes and potentially RIFE/frame interpolation nodes.
- For LTX 2.3, the Lightricks ComfyUI-LTXVideo nodes are likely necessary for advanced audio/video workflows.

## Current Runtime Ranking

1. `comfyui-sec-v89`: best optimized Comfy candidate because SageAttention/FlashAttention/xFormers/Triton import and Comfy starts with Sage Attention on RTX 5090.
2. `comfyui`: best clean stability baseline with newer Torch 2.12/cu130 and Comfy CUDA backend.
3. `direct-wan`: promising ownership path, but needs model-weight generation test and a decision on FlashAttention/source builds.

## Next Bake-Off Step

Download or locate WAN 2.2 weights, then run a minimal real generation in:

1. `comfyui-sec-v89` with Sage Attention.
2. `comfyui` clean runtime.
3. `direct-wan` using PyTorch attention fallback.

Capture:

- wall-clock time
- peak VRAM
- startup/model-load time
- output correctness
- workflow/API complexity
- LoRA/turbo path viability
