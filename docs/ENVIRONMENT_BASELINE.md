# Environment Baseline

Captured: 2026-05-26

## GPU

```text
GPU: NVIDIA GeForce RTX 5090
Driver: 591.86
CUDA runtime reported by nvidia-smi: 13.1
VRAM: 32607 MiB
Driver model: WDDM
```

At capture time, about 27.5 GB of VRAM was in use. One active compute process was `koboldcpp.exe`; several desktop/browser processes were also using the GPU. For accurate benchmarking, close unrelated GPU workloads before running video generation tests.

## CUDA

```text
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_PATH_V12_4=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
CUDA_PATH_V11_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
nvcc: 12.8.61
```

CUDA Toolkit 12.8 is the active development toolkit. The installed driver reports support for CUDA runtime 13.1, which is newer than the active toolkit and should be compatible with CUDA 12.8-built PyTorch wheels.

## Python

```text
python: 3.10.11 at D:\AI_STUFF\Python310\python.exe
py launcher: 3.13.9
available installs: 3.10, 3.11, 3.12, 3.13
```

Initial recommendation: use Python 3.10 or 3.11 for AI runtimes unless a model stack explicitly supports newer Python versions.

## Node

```text
node: 24.11.1
npm: 11.6.2
```

