from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wan_ltx_studio.rendering.profiles import get_renderer_profile
from wan_ltx_studio.rendering.runner_config import SegmentRunnerConfig


MEANINGFUL_TEST_FRAME_COUNT = 81


class SegmentRunnerError(RuntimeError):
    """Raised when direct segment execution fails."""


def main() -> None:
    args = _parse_args()
    config = SegmentRunnerConfig(
        profile_id=args.profile_id,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frame_num=args.frame_num,
        fps=args.fps,
        seed=args.seed,
        start_image=args.start_image,
        output_path=args.output,
        model_root=args.model_root,
        runtime_root=args.runtime_root,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        offload_model=args.offload_model,
        t5_cpu=args.t5_cpu,
        dry_run=args.dry_run,
        allow_gpu=args.allow_gpu,
        lock_path=args.lock_path,
        vae_tile_height=args.vae_tile_height,
        vae_tile_width=args.vae_tile_width,
        vae_stride_height=args.vae_stride_height,
        vae_stride_width=args.vae_stride_width,
    )
    payload = run_segment(config)
    print(json.dumps(payload, indent=2))


def run_segment(config: SegmentRunnerConfig) -> dict[str, Any]:
    profile = get_renderer_profile(config.profile_id)
    payload: dict[str, Any] = {
        "ok": False,
        "profileId": config.profile_id,
        "task": profile.task,
        "dryRun": config.dry_run,
        "allowGpu": config.allow_gpu,
        "outputPath": str(Path(config.output_path).resolve()),
        "config": asdict(config),
    }
    warnings = _runner_warnings(config)
    if warnings:
        payload["warnings"] = warnings

    if profile.id not in {
        "wan22_ti2v_5b_fp16",
        "wan22_i2v_a14b_fp8_original",
        "wan22_i2v_a14b_fp8_lightning_workflow",
    }:
        raise SegmentRunnerError(
            f"profile {profile.id} is not executable yet"
        )

    model_root = Path(config.model_root).resolve()
    runtime_root = Path(config.runtime_root).resolve()
    components = {component.role: model_root / component.relative_path for component in profile.all_components}
    missing = [str(path) for path in components.values() if not path.is_file()]
    if missing:
        raise SegmentRunnerError(f"missing required model files: {missing}")

    payload["components"] = {role: str(path) for role, path in components.items()}
    payload["runtimeRoot"] = str(runtime_root)

    if config.dry_run:
        payload["ok"] = True
        payload["message"] = "dry run only; no CUDA context or model load was started"
        return payload

    if not config.allow_gpu:
        raise SegmentRunnerError("GPU execution requires --allow-gpu")

    with _single_gpu_job_lock(Path(config.lock_path)):
        started = time.perf_counter()
        if profile.id == "wan22_ti2v_5b_fp16":
            telemetry = _run_ti2v_5b_segment(config, runtime_root, components)
        else:
            telemetry = _run_i2v_a14b_segment(config, runtime_root, components)
        payload["ok"] = True
        payload["elapsedSeconds"] = round(time.perf_counter() - started, 3)
        payload["telemetry"] = telemetry
        payload["message"] = "segment render complete"
        return payload


def _run_ti2v_5b_segment(
    config: SegmentRunnerConfig,
    runtime_root: Path,
    components: dict[str, Path],
) -> dict[str, Any]:
    sys.path.insert(0, str(runtime_root))

    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from easydict import EasyDict
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.textimage2video import WanTI2V
    from wan.utils.utils import save_video

    if not torch.cuda.is_available():
        raise SegmentRunnerError("CUDA is not available")

    output_path = Path(config.output_path).resolve()
    telemetry_path = output_path.with_suffix(".telemetry.jsonl")
    windows_gpu_memory_path = output_path.with_suffix(".windows-gpu-memory.jsonl")
    windows_gpu_monitor = _WindowsGpuMemoryMonitor(windows_gpu_memory_path)

    windows_gpu_monitor.start()
    try:
        return _run_ti2v_5b_segment_inner(
            config,
            components,
            output_path,
            telemetry_path,
            windows_gpu_memory_path,
            torch,
            Image,
            EasyDict,
            WAN_CONFIGS,
            WanModel,
            T5EncoderModel,
            Wan2_2_VAE,
            WanTI2V,
            save_video,
        )
    finally:
        windows_gpu_monitor.stop()


def _run_i2v_a14b_segment(
    config: SegmentRunnerConfig,
    runtime_root: Path,
    components: dict[str, Path],
) -> dict[str, Any]:
    sys.path.insert(0, str(runtime_root))

    import torch
    from PIL import Image
    from easydict import EasyDict
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.utils import save_video

    if not torch.cuda.is_available():
        raise SegmentRunnerError("CUDA is not available")

    output_path = Path(config.output_path).resolve()
    telemetry_path = output_path.with_suffix(".telemetry.jsonl")
    windows_gpu_memory_path = output_path.with_suffix(".windows-gpu-memory.jsonl")
    windows_gpu_monitor = _WindowsGpuMemoryMonitor(windows_gpu_memory_path)

    windows_gpu_monitor.start()
    try:
        return _run_i2v_a14b_segment_inner(
            config,
            components,
            output_path,
            telemetry_path,
            windows_gpu_memory_path,
            torch,
            Image,
            EasyDict,
            WAN_CONFIGS,
            WanModel,
            T5EncoderModel,
            Wan2_1_VAE,
            WanI2V,
            save_video,
        )
    finally:
        windows_gpu_monitor.stop()


def _run_ti2v_5b_segment_inner(
    config: SegmentRunnerConfig,
    components: dict[str, Path],
    output_path: Path,
    telemetry_path: Path,
    windows_gpu_memory_path: Path,
    torch,
    Image,
    EasyDict,
    WAN_CONFIGS,
    WanModel,
    T5EncoderModel,
    Wan2_2_VAE,
    WanTI2V,
    save_video,
) -> dict[str, Any]:
    _install_attention_fallback()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    telemetry = _CudaMemoryTelemetry(torch, telemetry_path)
    telemetry.mark(
        "cuda_ready",
        {
            "profileId": config.profile_id,
            "windowsGpuMemoryPath": str(windows_gpu_memory_path),
        },
    )

    cfg = EasyDict(WAN_CONFIGS["ti2v-5B"])
    cfg.t5_checkpoint = str(components["text_encoder"])
    cfg.vae_checkpoint = str(components["vae"])
    cfg.t5_dtype = torch.float16
    cfg.sample_steps = config.sample_steps
    cfg.sample_shift = config.sample_shift
    cfg.sample_guide_scale = config.sample_guide_scale
    cfg.sample_fps = int(config.fps)

    pipeline = WanTI2V.__new__(WanTI2V)
    pipeline.device = torch.device("cuda:0")
    pipeline.config = cfg
    pipeline.rank = 0
    pipeline.t5_cpu = config.t5_cpu
    pipeline.init_on_cpu = True
    pipeline.num_train_timesteps = cfg.num_train_timesteps
    pipeline.param_dtype = cfg.param_dtype
    pipeline.vae_stride = cfg.vae_stride
    pipeline.patch_size = cfg.patch_size
    pipeline.sp_size = 1
    pipeline.sample_neg_prompt = cfg.sample_neg_prompt
    telemetry.mark(
        "pipeline_shell_created",
        {
            "frameNum": config.frame_num,
            "width": config.width,
            "height": config.height,
            "sampleSteps": config.sample_steps,
        },
    )

    pipeline.text_encoder = _load_wan_t5_encoder(T5EncoderModel, cfg, components["text_encoder"])
    telemetry.mark("text_encoder_loaded_cpu")
    pipeline.vae = _load_wan22_vae(Wan2_2_VAE, components["vae"], pipeline.device)
    telemetry.mark("vae_loaded_cuda")
    _install_streaming_vae_decode(pipeline.vae, pipeline, telemetry, torch, config)
    telemetry.mark("vae_streaming_tiled_decode_installed")
    pipeline.model = _load_ti2v_model(WanModel, cfg, components["dit"])
    telemetry.mark("dit_loaded_cpu")
    _install_component_memory_hooks(pipeline, telemetry, expected_dit_forwards=config.sample_steps * 2)

    generation_width, generation_height = _generation_size_for_request(config, pipeline)
    img = Image.open(config.start_image).convert("RGB") if config.start_image else None
    if img is not None and (generation_width, generation_height) != (config.width, config.height):
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img = img.resize((generation_width, generation_height), resample)
        telemetry.mark(
            "start_image_resized_for_generation_grid",
            {
                "requestedWidth": config.width,
                "requestedHeight": config.height,
                "generationWidth": generation_width,
                "generationHeight": generation_height,
            },
        )
    telemetry.mark(
        "start_image_loaded_cpu" if img is not None else "no_start_image",
        {"startImage": config.start_image},
    )
    telemetry.mark("before_generate", synchronize=True)
    video = pipeline.generate(
        config.prompt,
        img=img,
        size=(generation_width, generation_height),
        max_area=generation_width * generation_height,
        frame_num=config.frame_num,
        shift=config.sample_shift,
        sample_solver="unipc",
        sampling_steps=config.sample_steps,
        guide_scale=config.sample_guide_scale,
        n_prompt=config.negative_prompt,
        seed=-1 if config.seed is None else config.seed,
        offload_model=config.offload_model,
    )
    video, crop_detail = _crop_or_resize_video_to_request(video, config.height, config.width, torch)
    if crop_detail is not None:
        telemetry.mark("video_matched_requested_size", crop_detail, synchronize=True)
    telemetry.mark("after_generate", synchronize=True)
    _offload_vae_after_decode(pipeline.vae, telemetry, torch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.mark("before_save_video")
    save_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=int(config.fps),
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    telemetry.mark("after_save_video", {"outputPath": str(output_path), "outputBytes": output_path.stat().st_size})

    telemetry.mark("before_cleanup", synchronize=True)
    del video, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    telemetry.mark("after_cleanup", synchronize=True)

    summary = telemetry.summary()
    summary["outputBytes"] = output_path.stat().st_size if output_path.exists() else None
    summary["telemetryPath"] = str(telemetry_path)
    summary["windowsGpuMemoryPath"] = str(windows_gpu_memory_path)
    return summary


def _run_i2v_a14b_segment_inner(
    config: SegmentRunnerConfig,
    components: dict[str, Path],
    output_path: Path,
    telemetry_path: Path,
    windows_gpu_memory_path: Path,
    torch,
    Image,
    EasyDict,
    WAN_CONFIGS,
    WanModel,
    T5EncoderModel,
    Wan2_1_VAE,
    WanI2V,
    save_video,
) -> dict[str, Any]:
    _install_attention_fallback()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    telemetry = _CudaMemoryTelemetry(torch, telemetry_path)
    telemetry.mark(
        "cuda_ready",
        {
            "profileId": config.profile_id,
            "windowsGpuMemoryPath": str(windows_gpu_memory_path),
        },
    )

    if not config.start_image:
        raise SegmentRunnerError("WAN A14B I2V requires a start image")

    cfg = EasyDict(WAN_CONFIGS["i2v-A14B"])
    cfg.t5_checkpoint = str(components["text_encoder"])
    cfg.vae_checkpoint = str(components["vae"])
    cfg.t5_dtype = torch.bfloat16
    cfg.param_dtype = torch.bfloat16
    cfg.sample_steps = config.sample_steps
    cfg.sample_shift = config.sample_shift
    cfg.sample_guide_scale = config.sample_guide_scale
    cfg.sample_fps = int(config.fps)

    pipeline = WanI2V.__new__(WanI2V)
    pipeline.device = torch.device("cuda:0")
    pipeline.config = cfg
    pipeline.rank = 0
    pipeline.t5_cpu = config.t5_cpu
    pipeline.init_on_cpu = True
    pipeline.num_train_timesteps = cfg.num_train_timesteps
    pipeline.boundary = cfg.boundary
    pipeline.param_dtype = cfg.param_dtype
    pipeline.vae_stride = cfg.vae_stride
    pipeline.patch_size = cfg.patch_size
    pipeline.sp_size = 1
    pipeline.sample_neg_prompt = cfg.sample_neg_prompt
    telemetry.mark(
        "pipeline_shell_created",
        {
            "frameNum": config.frame_num,
            "width": config.width,
            "height": config.height,
            "sampleSteps": config.sample_steps,
        },
    )

    pipeline.text_encoder = _load_wan_t5_encoder(
        T5EncoderModel,
        cfg,
        components["text_encoder"],
        torch_module=torch,
        fp8_scaled=True,
    )
    telemetry.mark("text_encoder_loaded_cpu")
    pipeline.vae = _load_wan21_vae(Wan2_1_VAE, components["vae"], pipeline.device, torch)
    telemetry.mark("vae_loaded_cuda")
    _install_streaming_vae_decode(
        pipeline.vae,
        pipeline,
        telemetry,
        torch,
        config,
        unpatchify_fn=lambda decoded, patch_size: decoded,
        spatial_upsample=8,
        decoder_accepts_first_chunk=False,
    )
    telemetry.mark("vae_streaming_tiled_decode_installed")

    pipeline.low_noise_model = _load_i2v_a14b_model(
        WanModel,
        cfg,
        components["low_noise_dit"],
        torch,
        lora_path=components.get("low_noise_lora"),
    )
    telemetry.mark("low_noise_dit_loaded_cpu")
    pipeline.high_noise_model = _load_i2v_a14b_model(
        WanModel,
        cfg,
        components["high_noise_dit"],
        torch,
        lora_path=components.get("high_noise_lora"),
    )
    telemetry.mark("high_noise_dit_loaded_cpu")
    _install_i2v_component_memory_hooks(pipeline, telemetry, expected_dit_forwards=config.sample_steps * 2)

    img = Image.open(config.start_image).convert("RGB")
    telemetry.mark("start_image_loaded_cpu", {"startImage": config.start_image})
    telemetry.mark("before_generate", synchronize=True)
    video = pipeline.generate(
        config.prompt,
        img=img,
        max_area=config.width * config.height,
        frame_num=config.frame_num,
        shift=config.sample_shift,
        sample_solver="unipc",
        sampling_steps=config.sample_steps,
        guide_scale=config.sample_guide_scale,
        n_prompt=config.negative_prompt,
        seed=-1 if config.seed is None else config.seed,
        offload_model=config.offload_model,
    )
    if tuple(video.shape[-2:]) != (config.height, config.width):
        raise SegmentRunnerError(
            f"A14B output size {tuple(video.shape[-2:])} did not match requested {(config.height, config.width)}"
        )
    telemetry.mark("after_generate", synchronize=True)
    _offload_vae_after_decode(pipeline.vae, telemetry, torch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.mark("before_save_video")
    save_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=int(config.fps),
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    telemetry.mark("after_save_video", {"outputPath": str(output_path), "outputBytes": output_path.stat().st_size})

    telemetry.mark("before_cleanup", synchronize=True)
    del video, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    telemetry.mark("after_cleanup", synchronize=True)

    summary = telemetry.summary()
    summary["outputBytes"] = output_path.stat().st_size if output_path.exists() else None
    summary["telemetryPath"] = str(telemetry_path)
    summary["windowsGpuMemoryPath"] = str(windows_gpu_memory_path)
    return summary


class _CudaMemoryTelemetry:
    def __init__(self, torch_module, sidecar_path: str | Path | None = None):
        self.torch = torch_module
        self.started = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self.sidecar_path = Path(sidecar_path).resolve() if sidecar_path else None
        if self.sidecar_path:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            self.sidecar_path.write_text("", encoding="utf-8")

    def mark(
        self,
        stage: str,
        detail: dict[str, Any] | None = None,
        *,
        synchronize: bool = False,
    ) -> dict[str, Any]:
        cuda = self.torch.cuda
        if synchronize:
            cuda.synchronize()

        allocated = cuda.memory_allocated()
        reserved = cuda.memory_reserved()
        peak_allocated = cuda.max_memory_allocated()
        peak_reserved = cuda.max_memory_reserved()
        snapshot: dict[str, Any] = {
            "stage": stage,
            "elapsedSeconds": round(time.perf_counter() - self.started, 3),
            "allocatedBytes": allocated,
            "reservedBytes": reserved,
            "peakAllocatedBytes": peak_allocated,
            "peakReservedBytes": peak_reserved,
            "allocatedGb": _bytes_to_gb(allocated),
            "reservedGb": _bytes_to_gb(reserved),
            "peakAllocatedGb": _bytes_to_gb(peak_allocated),
            "peakReservedGb": _bytes_to_gb(peak_reserved),
        }
        driver_snapshot = self._driver_memory_snapshot()
        if driver_snapshot:
            snapshot.update(driver_snapshot)
        if detail:
            snapshot["detail"] = detail
        self.stages.append(snapshot)
        self._write_sidecar(snapshot)
        return snapshot

    def summary(self) -> dict[str, Any]:
        cuda = self.torch.cuda
        current_allocated = cuda.memory_allocated()
        current_reserved = cuda.memory_reserved()
        peak_allocated = cuda.max_memory_allocated()
        peak_reserved = cuda.max_memory_reserved()
        driver_peaks = [stage.get("driverUsedBytes") for stage in self.stages if stage.get("driverUsedBytes") is not None]
        summary: dict[str, Any] = {
            "peakAllocatedBytes": peak_allocated,
            "peakReservedBytes": peak_reserved,
            "currentAllocatedBytes": current_allocated,
            "currentReservedBytes": current_reserved,
            "peakAllocatedGb": _bytes_to_gb(peak_allocated),
            "peakReservedGb": _bytes_to_gb(peak_reserved),
            "currentAllocatedGb": _bytes_to_gb(current_allocated),
            "currentReservedGb": _bytes_to_gb(current_reserved),
            "stages": self.stages,
        }
        if driver_peaks:
            peak_driver = max(driver_peaks)
            summary["peakDriverUsedBytes"] = peak_driver
            summary["peakDriverUsedGb"] = _bytes_to_gb(peak_driver)
        if self.sidecar_path:
            summary["telemetryPath"] = str(self.sidecar_path)
        return summary

    def _driver_memory_snapshot(self) -> dict[str, Any]:
        try:
            free_bytes, total_bytes = self.torch.cuda.mem_get_info()
        except (AttributeError, RuntimeError):
            return {}

        used_bytes = total_bytes - free_bytes
        return {
            "driverFreeBytes": free_bytes,
            "driverTotalBytes": total_bytes,
            "driverUsedBytes": used_bytes,
            "driverFreeGb": _bytes_to_gb(free_bytes),
            "driverTotalGb": _bytes_to_gb(total_bytes),
            "driverUsedGb": _bytes_to_gb(used_bytes),
        }

    def _write_sidecar(self, snapshot: dict[str, Any]) -> None:
        if not self.sidecar_path:
            return
        with self.sidecar_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")


def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


class _WindowsGpuMemoryMonitor:
    COUNTERS = (
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        r"\GPU Adapter Memory(*)\Shared Usage",
        r"\GPU Adapter Memory(*)\Total Committed",
    )

    def __init__(self, output_path: str | Path, interval_seconds: float = 1.0):
        self.output_path = Path(output_path).resolve()
        self.interval_seconds = interval_seconds
        self.started = time.perf_counter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if platform.system() != "Windows":
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="windows-gpu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = _sample_windows_gpu_memory_once(self.COUNTERS)
                sample["elapsedSeconds"] = round(time.perf_counter() - self.started, 3)
                sample["source"] = "typeperf"
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
            except Exception as exc:
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "elapsedSeconds": round(time.perf_counter() - self.started, 3),
                                "error": str(exc),
                                "source": "typeperf",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                return
            self._stop_event.wait(self.interval_seconds)


def _sample_windows_gpu_memory_once(counters: tuple[str, ...]) -> dict[str, Any]:
    result = subprocess.run(
        ["typeperf", *counters, "-sc", "1"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return _parse_typeperf_gpu_memory_sample(result.stdout)


def _parse_typeperf_gpu_memory_sample(output: str) -> dict[str, Any]:
    rows = [row for row in csv.reader(output.splitlines()) if row]
    header_index = next(
        (index for index, row in enumerate(rows) if row[0].startswith("(PDH-CSV")),
        None,
    )
    if header_index is None:
        raise SegmentRunnerError("typeperf GPU memory output did not include a PDH header")

    header = rows[header_index]
    data = next(
        (row for row in rows[header_index + 1 :] if len(row) == len(header) and _looks_like_typeperf_data_row(row)),
        None,
    )
    if data is None:
        raise SegmentRunnerError("typeperf GPU memory output did not include a data row")

    dedicated = 0.0
    shared = 0.0
    committed = 0.0
    for name, raw_value in zip(header[1:], data[1:]):
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if name.endswith(r"\Dedicated Usage"):
            dedicated += value
        elif name.endswith(r"\Shared Usage"):
            shared += value
        elif name.endswith(r"\Total Committed"):
            committed += value

    return {
        "timestamp": data[0],
        "windowsDedicatedUsageBytes": int(dedicated),
        "windowsSharedUsageBytes": int(shared),
        "windowsTotalCommittedBytes": int(committed),
        "windowsDedicatedUsageGb": _bytes_to_gb(dedicated),
        "windowsSharedUsageGb": _bytes_to_gb(shared),
        "windowsTotalCommittedGb": _bytes_to_gb(committed),
    }


def _looks_like_typeperf_data_row(row: list[str]) -> bool:
    return len(row) > 1 and any(_is_float(value) for value in row[1:])


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _install_streaming_vae_decode(
    vae,
    pipeline,
    telemetry: _CudaMemoryTelemetry,
    torch_module,
    config: SegmentRunnerConfig,
    *,
    unpatchify_fn=None,
    spatial_upsample: int = 16,
    decoder_accepts_first_chunk: bool = True,
) -> None:
    import types
    import wan.modules.vae2_2 as vae2_2

    if unpatchify_fn is None:
        unpatchify_fn = vae2_2.unpatchify

    def streaming_decode(self, zs):
        if not isinstance(zs, list):
            raise TypeError("zs should be a list")
        _offload_non_vae_components_before_decode(pipeline, telemetry, torch_module)
        with _vae_autocast_context(torch_module, self):
            return [
                _decode_wan22_vae_tiled_stream(
                    self,
                    latent,
                    unpatchify_fn=unpatchify_fn,
                    telemetry=telemetry,
                    torch_module=torch_module,
                    video_index=video_index,
                    tile_size=(config.vae_tile_height, config.vae_tile_width),
                    tile_stride=(config.vae_stride_height, config.vae_stride_width),
                    spatial_upsample=spatial_upsample,
                    decoder_accepts_first_chunk=decoder_accepts_first_chunk,
                )
                for video_index, latent in enumerate(zs)
            ]

    vae.decode = types.MethodType(streaming_decode, vae)


def _offload_non_vae_components_before_decode(pipeline, telemetry: _CudaMemoryTelemetry, torch_module) -> None:
    telemetry.mark("pre_vae_decode_offload_start")
    if getattr(pipeline, "model", None) is not None:
        pipeline.model.cpu()
    for attr in ("low_noise_model", "high_noise_model"):
        model = getattr(pipeline, attr, None)
        if model is not None:
            model.cpu()
    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None and getattr(text_encoder, "model", None) is not None:
        text_encoder.model.cpu()
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()
    telemetry.mark("pre_vae_decode_offload_end", synchronize=torch_module.cuda.is_available())


def _offload_vae_after_decode(vae, telemetry: _CudaMemoryTelemetry, torch_module) -> None:
    telemetry.mark("post_vae_decode_offload_start")
    if getattr(vae, "model", None) is not None:
        vae.model.cpu()
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()
    telemetry.mark("post_vae_decode_offload_end", synchronize=torch_module.cuda.is_available())


def _vae_autocast_context(torch_module, vae):
    device = vae.device
    device_type = getattr(device, "type", str(device))
    if device_type != "cuda":
        return nullcontext()
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        return torch_module.amp.autocast("cuda", dtype=vae.dtype)
    return torch_module.cuda.amp.autocast(dtype=vae.dtype)


def _generation_size_for_request(config: SegmentRunnerConfig, pipeline) -> tuple[int, int]:
    height_quantum = _spatial_generation_quantum(pipeline, axis=1)
    width_quantum = _spatial_generation_quantum(pipeline, axis=2)
    return _ceil_to_multiple(config.width, width_quantum), _ceil_to_multiple(config.height, height_quantum)


def _spatial_generation_quantum(pipeline, *, axis: int) -> int:
    patch_size = getattr(pipeline, "patch_size", (1, 1, 1))
    vae_stride = getattr(pipeline, "vae_stride", (1, 16, 16))
    return max(1, int(patch_size[axis]) * int(vae_stride[axis]))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _crop_or_resize_video_to_request(video, target_height: int, target_width: int, torch_module):
    current_height = int(video.shape[-2])
    current_width = int(video.shape[-1])
    if (current_height, current_width) == (target_height, target_width):
        return video, None

    detail = {
        "sourceWidth": current_width,
        "sourceHeight": current_height,
        "targetWidth": target_width,
        "targetHeight": target_height,
    }
    if current_height >= target_height and current_width >= target_width:
        top = (current_height - target_height) // 2
        left = (current_width - target_width) // 2
        detail.update({"mode": "center_crop", "cropTop": top, "cropLeft": left})
        return video[..., top : top + target_height, left : left + target_width].contiguous(), detail

    frames = video.permute(1, 0, 2, 3)
    resized = torch_module.nn.functional.interpolate(
        frames,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    detail["mode"] = "resize"
    return resized.permute(1, 0, 2, 3).contiguous(), detail


def _decode_wan22_vae_tiled_stream(
    vae,
    latent,
    *,
    unpatchify_fn,
    telemetry: _CudaMemoryTelemetry | None,
    torch_module,
    video_index: int = 0,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
    spatial_upsample: int = 16,
    decoder_accepts_first_chunk: bool = True,
):
    latent_height = int(latent.shape[-2])
    latent_width = int(latent.shape[-1])
    tile_height = max(1, min(int(tile_size[0]), latent_height))
    tile_width = max(1, min(int(tile_size[1]), latent_width))
    stride_height = max(1, min(int(tile_stride[0]), tile_height))
    stride_width = max(1, min(int(tile_stride[1]), tile_width))
    tasks = _spatial_tile_tasks(latent_height, latent_width, tile_height, tile_width, stride_height, stride_width)

    if len(tasks) == 1:
        return _decode_wan22_vae_temporal_stream(
            vae,
            latent,
            unpatchify_fn=unpatchify_fn,
            telemetry=telemetry,
            torch_module=torch_module,
            video_index=video_index,
            decoder_accepts_first_chunk=decoder_accepts_first_chunk,
        )

    latent_is_cuda = getattr(latent, "is_cuda", False)
    if telemetry:
        telemetry.mark(
            "vae_tiled_decode_start",
            {
                "videoIndex": video_index,
                "latentHeight": latent_height,
                "latentWidth": latent_width,
                "tileHeight": tile_height,
                "tileWidth": tile_width,
                "strideHeight": stride_height,
                "strideWidth": stride_width,
                "tiles": len(tasks),
            },
            synchronize=latent_is_cuda,
        )

    values = None
    weights = None
    overlap_height = max(0, (tile_height - stride_height) * spatial_upsample)
    overlap_width = max(0, (tile_width - stride_width) * spatial_upsample)

    for tile_index, (top, bottom, left, right) in enumerate(tasks):
        if telemetry:
            telemetry.mark(
                "vae_tiled_decode_tile_start",
                {
                    "videoIndex": video_index,
                    "tile": tile_index + 1,
                    "tiles": len(tasks),
                    "latentTop": top,
                    "latentBottom": bottom,
                    "latentLeft": left,
                    "latentRight": right,
                },
                synchronize=latent_is_cuda,
            )

        latent_tile = latent[:, :, top:bottom, left:right].contiguous()
        decoded_tile = _decode_wan22_vae_temporal_stream(
            vae,
            latent_tile,
            unpatchify_fn=unpatchify_fn,
            telemetry=None,
            torch_module=torch_module,
            video_index=video_index,
            decoder_accepts_first_chunk=decoder_accepts_first_chunk,
        )

        if values is None:
            output_height = latent_height * spatial_upsample
            output_width = latent_width * spatial_upsample
            values = torch_module.zeros(
                (decoded_tile.shape[0], decoded_tile.shape[1], output_height, output_width),
                dtype=decoded_tile.dtype,
                device="cpu",
            )
            weights = torch_module.zeros(
                (1, 1, output_height, output_width),
                dtype=decoded_tile.dtype,
                device="cpu",
            )

        target_top = top * spatial_upsample
        target_left = left * spatial_upsample
        tile_output_height = min(int(decoded_tile.shape[-2]), int(values.shape[-2]) - target_top)
        tile_output_width = min(int(decoded_tile.shape[-1]), int(values.shape[-1]) - target_left)
        decoded_tile = decoded_tile[:, :, :tile_output_height, :tile_output_width]

        mask = _spatial_feather_mask(
            torch_module,
            height=tile_output_height,
            width=tile_output_width,
            top_bound=top == 0,
            bottom_bound=bottom >= latent_height,
            left_bound=left == 0,
            right_bound=right >= latent_width,
            overlap_height=overlap_height,
            overlap_width=overlap_width,
            dtype=decoded_tile.dtype,
        )

        target_bottom = target_top + tile_output_height
        target_right = target_left + tile_output_width
        values[:, :, target_top:target_bottom, target_left:target_right].add_(decoded_tile * mask)
        weights[:, :, target_top:target_bottom, target_left:target_right].add_(mask)

        del latent_tile, decoded_tile, mask
        _empty_cuda_cache_for_tensor(torch_module, latent)

        if telemetry and _should_mark_vae_tile(tile_index + 1, len(tasks)):
            telemetry.mark(
                "vae_tiled_decode_tile_cpu",
                {
                    "videoIndex": video_index,
                    "tile": tile_index + 1,
                    "tiles": len(tasks),
                },
                synchronize=latent_is_cuda,
            )

    if values is None or weights is None:
        raise SegmentRunnerError("VAE tiled decode produced no tiles")

    output = values / weights.clamp_min(1e-6)
    output.clamp_(-1, 1)
    if telemetry:
        telemetry.mark("vae_tiled_decode_end", {"videoIndex": video_index, "tiles": len(tasks)}, synchronize=latent_is_cuda)
    return output


def _spatial_tile_tasks(
    latent_height: int,
    latent_width: int,
    tile_height: int,
    tile_width: int,
    stride_height: int,
    stride_width: int,
) -> list[tuple[int, int, int, int]]:
    tasks = []
    for top in range(0, latent_height, stride_height):
        if top - stride_height >= 0 and top - stride_height + tile_height >= latent_height:
            continue
        for left in range(0, latent_width, stride_width):
            if left - stride_width >= 0 and left - stride_width + tile_width >= latent_width:
                continue
            tasks.append(
                (
                    top,
                    min(top + tile_height, latent_height),
                    left,
                    min(left + tile_width, latent_width),
                )
            )
    return tasks


def _spatial_feather_mask(
    torch_module,
    *,
    height: int,
    width: int,
    top_bound: bool,
    bottom_bound: bool,
    left_bound: bool,
    right_bound: bool,
    overlap_height: int,
    overlap_width: int,
    dtype,
):
    h = _feather_axis_mask(
        torch_module,
        height,
        leading_bound=top_bound,
        trailing_bound=bottom_bound,
        overlap=overlap_height,
        dtype=dtype,
    )
    w = _feather_axis_mask(
        torch_module,
        width,
        leading_bound=left_bound,
        trailing_bound=right_bound,
        overlap=overlap_width,
        dtype=dtype,
    )
    return torch_module.minimum(h.view(1, 1, height, 1), w.view(1, 1, 1, width))


def _feather_axis_mask(torch_module, length: int, *, leading_bound: bool, trailing_bound: bool, overlap: int, dtype):
    mask = torch_module.ones((length,), dtype=dtype, device="cpu")
    overlap = max(0, min(int(overlap), length))
    if overlap == 0:
        return mask
    ramp = torch_module.arange(1, overlap + 1, dtype=dtype, device="cpu") / overlap
    if not leading_bound:
        mask[:overlap] = torch_module.minimum(mask[:overlap], ramp)
    if not trailing_bound:
        mask[-overlap:] = torch_module.minimum(mask[-overlap:], torch_module.flip(ramp, dims=(0,)))
    return mask


def _decode_wan22_vae_temporal_stream(
    vae,
    latent,
    *,
    unpatchify_fn,
    telemetry: _CudaMemoryTelemetry | None,
    torch_module,
    video_index: int = 0,
    decoder_accepts_first_chunk: bool = True,
):
    model = vae.model
    model.clear_cache()
    latent_is_cuda = getattr(latent, "is_cuda", False)
    try:
        z = latent.unsqueeze(0)
        if hasattr(vae.scale[0], "view"):
            z = z / vae.scale[1].view(1, model.z_dim, 1, 1, 1) + vae.scale[0].view(
                1, model.z_dim, 1, 1, 1
            )
        else:
            z = z / vae.scale[1] + vae.scale[0]

        x = model.conv2(z)
        total_chunks = int(x.shape[2])
        decoded_chunks = []
        if telemetry:
            telemetry.mark(
                "vae_temporal_decode_conv2_end",
                {"videoIndex": video_index, "chunks": total_chunks},
                synchronize=latent_is_cuda,
            )

        for chunk_index in range(total_chunks):
            model._conv_idx = [0]
            decoder_kwargs = {
                "feat_cache": model._feat_map,
                "feat_idx": model._conv_idx,
            }
            if decoder_accepts_first_chunk and chunk_index == 0:
                decoder_kwargs["first_chunk"] = True

            decoded_gpu = model.decoder(
                x[:, :, chunk_index : chunk_index + 1, :, :],
                **decoder_kwargs,
            )
            decoded_gpu = unpatchify_fn(decoded_gpu, patch_size=2)
            decoded_cpu = decoded_gpu.float().clamp_(-1, 1).squeeze(0).cpu()
            decoded_chunks.append(decoded_cpu)
            del decoded_gpu
            _empty_cuda_cache_for_tensor(torch_module, latent)

            if telemetry and _should_mark_vae_chunk(chunk_index + 1, total_chunks):
                telemetry.mark(
                    "vae_temporal_decode_chunk_cpu",
                    {
                        "videoIndex": video_index,
                        "chunk": chunk_index + 1,
                        "chunks": total_chunks,
                        "chunkFrames": int(decoded_cpu.shape[1]),
                        "cpuChunksHeld": len(decoded_chunks),
                    },
                    synchronize=latent_is_cuda,
                )

        return torch_module.cat(decoded_chunks, dim=1)
    finally:
        model.clear_cache()
        _empty_cuda_cache_for_tensor(torch_module, latent)


def _empty_cuda_cache_for_tensor(torch_module, tensor) -> None:
    if getattr(tensor, "is_cuda", False):
        torch_module.cuda.empty_cache()


def _should_mark_vae_chunk(chunk_number: int, total_chunks: int) -> bool:
    return chunk_number <= 2 or chunk_number == total_chunks or chunk_number % 4 == 0


def _should_mark_vae_tile(tile_number: int, total_tiles: int) -> bool:
    return tile_number <= 2 or tile_number == total_tiles or tile_number % 4 == 0


def _runner_warnings(config: SegmentRunnerConfig) -> list[str]:
    warnings = []
    if config.frame_num != MEANINGFUL_TEST_FRAME_COUNT:
        warnings.append(
            f"{config.frame_num} frames is a smoke/preview run; use {MEANINGFUL_TEST_FRAME_COUNT} frames for meaningful WAN calibration."
        )
    return warnings


def _install_component_memory_hooks(pipeline, telemetry: _CudaMemoryTelemetry, expected_dit_forwards: int) -> None:
    _wrap_method(pipeline.text_encoder.model, "to", telemetry, "text_encoder_to_device")
    _wrap_method(pipeline.text_encoder.model, "cpu", telemetry, "text_encoder_to_cpu")
    _wrap_counted_method(
        pipeline.text_encoder.model,
        "forward",
        telemetry,
        "text_encoder_forward",
        expected_calls=2,
        mark_every=1,
    )
    _wrap_method(pipeline.vae, "encode", telemetry, "vae_encode")
    _wrap_method(pipeline.vae, "decode", telemetry, "vae_decode")
    _wrap_method(pipeline.model, "to", telemetry, "dit_to_device")
    _wrap_method(pipeline.model, "cpu", telemetry, "dit_to_cpu")
    _wrap_counted_method(
        pipeline.model,
        "forward",
        telemetry,
        "dit_forward",
        expected_calls=expected_dit_forwards,
        mark_every=10,
    )


def _install_i2v_component_memory_hooks(pipeline, telemetry: _CudaMemoryTelemetry, expected_dit_forwards: int) -> None:
    _wrap_method(pipeline.text_encoder.model, "to", telemetry, "text_encoder_to_device")
    _wrap_method(pipeline.text_encoder.model, "cpu", telemetry, "text_encoder_to_cpu")
    for attr, stage in (
        ("low_noise_model", "low_noise_dit"),
        ("high_noise_model", "high_noise_dit"),
    ):
        model = getattr(pipeline, attr)
        _wrap_method(model, "to", telemetry, f"{stage}_to_device")
        _wrap_method(model, "cpu", telemetry, f"{stage}_to_cpu")
        _wrap_counted_method(
            model,
            "forward",
            telemetry,
            f"{stage}_forward",
            expected_calls=expected_dit_forwards,
            mark_every=10,
        )
    _wrap_method(pipeline.vae, "encode", telemetry, "vae_encode")
    _wrap_method(pipeline.vae, "decode", telemetry, "vae_decode")


def _wrap_method(target, method_name: str, telemetry: _CudaMemoryTelemetry, stage: str) -> None:
    original = getattr(target, method_name)

    def wrapped(*args, **kwargs):
        telemetry.mark(f"{stage}_start")
        result = original(*args, **kwargs)
        telemetry.mark(f"{stage}_end", synchronize=True)
        return result

    setattr(target, method_name, wrapped)


def _wrap_counted_method(
    target,
    method_name: str,
    telemetry: _CudaMemoryTelemetry,
    stage: str,
    *,
    expected_calls: int,
    mark_every: int,
) -> None:
    original = getattr(target, method_name)
    state = {"calls": 0}

    def should_mark(call_index: int) -> bool:
        return call_index <= 2 or call_index == expected_calls or call_index % mark_every == 0

    def wrapped(*args, **kwargs):
        state["calls"] += 1
        call_index = state["calls"]
        if should_mark(call_index):
            telemetry.mark(
                f"{stage}_start",
                {"call": call_index, "expectedCalls": expected_calls},
            )
        result = original(*args, **kwargs)
        if should_mark(call_index):
            telemetry.mark(
                f"{stage}_end",
                {"call": call_index, "expectedCalls": expected_calls},
                synchronize=True,
            )
        return result

    setattr(target, method_name, wrapped)


def _load_ti2v_model(model_cls, cfg, path: Path):
    from safetensors.torch import load_file

    import torch

    with torch.device("meta"):
        model = model_cls(
            model_type="ti2v",
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=48,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            text_dim=4096,
            out_dim=48,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
        )
    state_dict = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise SegmentRunnerError(f"TI2V model load mismatch: missing={missing}, unexpected={unexpected}")
    _reset_wan_rope_freqs(model, cfg)
    return model.eval().requires_grad_(False)


def _reset_wan_rope_freqs(model, cfg) -> None:
    import torch

    head_dim = cfg.dim // cfg.num_heads
    model.freqs = torch.cat(
        [
            _rope_params(1024, head_dim - 4 * (head_dim // 6)),
            _rope_params(1024, 2 * (head_dim // 6)),
            _rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )


def _rope_params(max_seq_len: int, dim: int, theta: int = 10000):
    import torch

    if dim % 2 != 0:
        raise SegmentRunnerError(f"RoPE dimension must be even, got {dim}")
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _install_attention_fallback() -> None:
    import torch
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module

    if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE:
        return

    def sdpa_flash_attention_fallback(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del window_size, deterministic, version
        out_dtype = q.dtype

        def lens_values(lens, batch_size: int, fallback: int) -> list[int]:
            if lens is None:
                return [fallback] * batch_size
            if isinstance(lens, int):
                return [lens] * batch_size
            if hasattr(lens, "detach"):
                values = [int(item) for item in lens.detach().cpu().tolist()]
            else:
                values = [int(item) for item in lens]
            if len(values) != batch_size:
                raise SegmentRunnerError(
                    f"attention length metadata has {len(values)} items for batch size {batch_size}"
                )
            return values

        if q_scale is not None:
            q = q * q_scale

        half_dtypes = (torch.float16, torch.bfloat16)
        work_dtype = q.dtype if q.dtype in half_dtypes else dtype
        q = q.to(work_dtype)
        k = k.to(work_dtype)
        v = v.to(work_dtype if v.dtype not in half_dtypes else v.dtype)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        batch_size, query_len = q.shape[:2]
        key_len = k.shape[1]
        q_lengths = lens_values(q_lens, batch_size, query_len)
        k_lengths = lens_values(k_lens, batch_size, key_len)
        out = v.new_zeros(batch_size, query_len, q.shape[2], v.shape[-1])

        for batch_index, (current_q_len, current_k_len) in enumerate(zip(q_lengths, k_lengths)):
            current_q = q[batch_index : batch_index + 1, :current_q_len].transpose(1, 2)
            current_k = k[batch_index : batch_index + 1, :current_k_len].transpose(1, 2)
            current_v = v[batch_index : batch_index + 1, :current_k_len].transpose(1, 2)
            current_out = torch.nn.functional.scaled_dot_product_attention(
                current_q,
                current_k,
                current_v,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            out[batch_index : batch_index + 1, :current_q_len] = current_out.transpose(1, 2)

        return out.to(out_dtype).contiguous()

    attention_module.flash_attention = sdpa_flash_attention_fallback
    model_module.flash_attention = sdpa_flash_attention_fallback


def _load_wan22_vae(vae_cls, path: Path, device):
    import torch
    from safetensors.torch import load_file

    import wan.modules.vae2_2 as vae2_2

    original_video_vae = vae2_2._video_vae
    try:
        vae2_2._video_vae = lambda *args, **kwargs: torch.nn.Identity()
        vae = vae_cls(vae_pth=None, device=device)
    finally:
        vae2_2._video_vae = original_video_vae

    with torch.device("meta"):
        model = vae2_2.WanVAE_(
            dim=160,
            z_dim=48,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
    state_dict = _normalize_wan22_vae_state_dict(load_file(str(path), device="cpu"))
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise SegmentRunnerError(f"WAN 2.2 VAE load mismatch: missing={missing}, unexpected={unexpected}")
    vae.model = model.eval().requires_grad_(False).to(device)
    return vae


def _load_wan21_vae(vae_cls, path: Path, device, torch_module):
    from safetensors.torch import load_file

    import wan.modules.vae2_1 as vae2_1

    original_video_vae = vae2_1._video_vae
    try:
        vae2_1._video_vae = lambda *args, **kwargs: torch_module.nn.Identity()
        vae = vae_cls(vae_pth=None, dtype=torch_module.bfloat16, device=device)
    finally:
        vae2_1._video_vae = original_video_vae

    with torch_module.device("meta"):
        model = vae2_1.WanVAE_(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
    state_dict = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise SegmentRunnerError(f"WAN 2.1 VAE load mismatch: missing={missing}, unexpected={unexpected}")
    vae.model = model.eval().requires_grad_(False).to(device)
    return vae


def _load_i2v_a14b_model(model_cls, cfg, path: Path, torch_module, *, lora_path: Path | None = None):
    from safetensors.torch import load_file

    with torch_module.device("meta"):
        model = model_cls(
            model_type="i2v",
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=36,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            text_dim=4096,
            out_dim=16,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
        )
    raw_state_dict = load_file(str(path), device="cpu")
    state_dict, scale_weights = _split_fp8_scaled_state_dict(raw_state_dict)
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise SegmentRunnerError(f"A14B I2V model load mismatch: missing={missing}, unexpected={unexpected}")
    loras = _load_wan_lora_adapters(lora_path, torch_module) if lora_path is not None else {}
    _patch_fp8_scaled_linears(model, torch_module, cfg.param_dtype, scale_weights, loras=loras)
    _reset_wan_rope_freqs(model, cfg)
    return model.eval().requires_grad_(False)


def _split_fp8_scaled_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    weights = {}
    scale_weights = {}
    for key, value in state_dict.items():
        if key.endswith(".scale_weight"):
            scale_weights[key.removesuffix(".scale_weight")] = value
            continue
        if key.endswith(".scale_input") or key == "scaled_fp8":
            continue
        weights[key] = value
    return weights, scale_weights


def _load_wan_lora_adapters(path: Path, torch_module) -> dict[str, dict[str, Any]]:
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    adapters: dict[str, dict[str, Any]] = {}
    prefix = "diffusion_model."
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        key = key[len(prefix) :]
        for suffix, field in (
            (".lora_down.weight", "down"),
            (".lora_up.weight", "up"),
            (".alpha", "alpha"),
        ):
            if key.endswith(suffix):
                module_name = key.removesuffix(suffix)
                adapters.setdefault(module_name, {})[field] = value
                break
    return {
        name: adapter
        for name, adapter in adapters.items()
        if {"down", "up", "alpha"}.issubset(adapter)
    }


def _patch_fp8_scaled_linears(
    module,
    torch_module,
    base_dtype,
    scale_weights: dict[str, Any],
    *,
    loras: dict[str, dict[str, Any]] | None = None,
) -> int:
    patched = 0
    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch_module, "float8_e4m3fn", None),
            getattr(torch_module, "float8_e5m2", None),
        )
        if dtype is not None
    )
    for name, submodule in module.named_modules():
        if not isinstance(submodule, torch_module.nn.Linear):
            continue
        scale_weight = scale_weights.get(name)
        lora = (loras or {}).get(name)
        if scale_weight is not None:
            submodule.register_buffer("_wan_ltx_scale_weight", scale_weight.float(), persistent=False)
        if lora is not None:
            submodule.register_buffer("_wan_ltx_lora_down", lora["down"], persistent=False)
            submodule.register_buffer("_wan_ltx_lora_up", lora["up"], persistent=False)
            submodule.register_buffer("_wan_ltx_lora_alpha", lora["alpha"].float().reshape(()), persistent=False)
            submodule._wan_ltx_lora_strength = 1.0
        if submodule.weight.dtype in fp8_dtypes or lora is not None:
            submodule._wan_ltx_base_dtype = torch_module.float32 if name.startswith("time_") else base_dtype
            submodule.forward = types.MethodType(_fp8_scaled_linear_forward, submodule)
            patched += 1
    return patched


def _fp8_scaled_linear_forward(self, input):
    import torch

    weight = self.weight
    bias = self.bias
    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    )
    base_dtype = getattr(self, "_wan_ltx_base_dtype", input.dtype)
    if weight.dtype in fp8_dtypes:
        out = _fp8_scaled_linear(input, weight, bias, getattr(self, "_wan_ltx_scale_weight", None), base_dtype, torch)
    else:
        out = torch.nn.functional.linear(input, weight, bias)

    if hasattr(self, "_wan_ltx_lora_down"):
        lora_input = input.to(base_dtype)
        down = self._wan_ltx_lora_down.to(device=input.device, dtype=base_dtype)
        up = self._wan_ltx_lora_up.to(device=input.device, dtype=base_dtype)
        alpha = self._wan_ltx_lora_alpha.to(device=input.device, dtype=base_dtype)
        rank = max(1, int(down.shape[0]))
        strength = getattr(self, "_wan_ltx_lora_strength", 1.0)
        lora_out = torch.nn.functional.linear(torch.nn.functional.linear(lora_input, down), up)
        out = out + lora_out * (alpha / rank) * strength
    return out


def _fp8_scaled_linear(input, weight, bias, scale_weight, out_dtype, torch_module):
    if input.device.type == "cuda" and hasattr(torch_module, "_scaled_mm"):
        input_shape = input.shape
        scale_input = torch_module.ones((), device=input.device, dtype=torch_module.float32)
        scale_b = (
            scale_weight.to(device=input.device, dtype=torch_module.float32).squeeze()
            if scale_weight is not None
            else torch_module.ones((), device=input.device, dtype=torch_module.float32)
        )
        flat = input.clamp(min=-448, max=448).reshape(-1, input_shape[-1]).to(torch_module.float8_e4m3fn).contiguous()
        linear_bias = bias.to(out_dtype) if bias is not None else None
        out = torch_module._scaled_mm(
            flat,
            weight.t(),
            out_dtype=out_dtype,
            bias=linear_bias,
            scale_a=scale_input,
            scale_b=scale_b,
        )
        return out.reshape(*input_shape[:-1], weight.shape[0])

    work_weight = weight.to(out_dtype)
    if scale_weight is not None:
        work_weight = work_weight * scale_weight.to(device=work_weight.device, dtype=out_dtype)
    linear_bias = bias.to(out_dtype) if bias is not None else None
    return torch_module.nn.functional.linear(input.to(out_dtype), work_weight, linear_bias)


def _load_wan_t5_encoder(t5_cls, cfg, path: Path, *, torch_module=None, fp8_scaled: bool = False):
    import torch
    from safetensors.torch import load_file
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

    torch_module = torch_module or torch

    encoder = t5_cls.__new__(t5_cls)
    encoder.text_len = cfg.text_len
    encoder.dtype = cfg.t5_dtype
    encoder.device = torch.device("cpu")
    encoder.checkpoint_path = str(path)
    encoder.tokenizer_path = cfg.t5_tokenizer

    with torch.device("meta"):
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=cfg.t5_dtype,
            device="meta",
        ).eval().requires_grad_(False)
    raw_state_dict = load_file(str(path), device="cpu")
    state_dict, scale_weights = _convert_comfy_umt5_state_dict_with_scales(raw_state_dict)
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise SegmentRunnerError(f"UMT5 load mismatch: missing={missing}, unexpected={unexpected}")
    if fp8_scaled or scale_weights:
        _patch_fp8_scaled_linears(model, torch_module, cfg.t5_dtype, scale_weights)
    encoder.model = model.to(encoder.device)
    encoder.tokenizer = HuggingfaceTokenizer(
        name=cfg.t5_tokenizer,
        seq_len=cfg.text_len,
        clean="whitespace",
    )
    return encoder


def _convert_comfy_umt5_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    converted, _ = _convert_comfy_umt5_state_dict_with_scales(state_dict)
    return converted


def _convert_comfy_umt5_state_dict_with_scales(state_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    converted = {}
    scale_weights = {}
    for key, value in state_dict.items():
        if key == "spiece_model":
            continue
        if key.endswith(".scale_weight"):
            converted_key = _convert_comfy_umt5_weight_key(key.removesuffix(".scale_weight") + ".weight")
            scale_weights[converted_key.removesuffix(".weight")] = value
            continue
        if key.endswith(".scale_input") or key == "scaled_fp8":
            continue
        converted[_convert_comfy_umt5_weight_key(key)] = value
    return converted, scale_weights


def _convert_comfy_umt5_weight_key(key: str) -> str:
    if key == "shared.weight":
        return "token_embedding.weight"
    if key == "encoder.final_layer_norm.weight":
        return "norm.weight"
    if key.startswith("encoder.block."):
        parts = key.split(".")
        block = parts[2]
        suffix = ".".join(parts[3:])
        prefix = f"blocks.{block}"
        replacements = {
            "layer.0.layer_norm.weight": f"{prefix}.norm1.weight",
            "layer.0.SelfAttention.q.weight": f"{prefix}.attn.q.weight",
            "layer.0.SelfAttention.k.weight": f"{prefix}.attn.k.weight",
            "layer.0.SelfAttention.v.weight": f"{prefix}.attn.v.weight",
            "layer.0.SelfAttention.o.weight": f"{prefix}.attn.o.weight",
            "layer.0.SelfAttention.relative_attention_bias.weight": f"{prefix}.pos_embedding.embedding.weight",
            "layer.1.layer_norm.weight": f"{prefix}.norm2.weight",
            "layer.1.DenseReluDense.wi_0.weight": f"{prefix}.ffn.gate.0.weight",
            "layer.1.DenseReluDense.wi_1.weight": f"{prefix}.ffn.fc1.weight",
            "layer.1.DenseReluDense.wo.weight": f"{prefix}.ffn.fc2.weight",
        }
        if suffix in replacements:
            return replacements[suffix]
    raise SegmentRunnerError(f"unsupported UMT5 key: {key}")


def _normalize_wan22_vae_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return state_dict


@contextmanager
def _single_gpu_job_lock(lock_path: Path):
    lock_path = lock_path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    except FileExistsError as exc:
        raise SegmentRunnerError(f"render lock is already held: {lock_path}") from exc
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one direct WAN segment.")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frame-num", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--start-image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--runtime-root", default="runtimes/direct-wan/src/Wan2.2")
    parser.add_argument("--sample-steps", type=int, required=True)
    parser.add_argument("--sample-shift", type=float, required=True)
    parser.add_argument("--sample-guide-scale", type=_parse_sample_guide_scale, required=True)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--lock-path", default="renders/.render.lock")
    parser.add_argument("--vae-tile-height", type=int, default=34)
    parser.add_argument("--vae-tile-width", type=int, default=34)
    parser.add_argument("--vae-stride-height", type=int, default=18)
    parser.add_argument("--vae-stride-width", type=int, default=16)
    return parser.parse_args()


def _parse_sample_guide_scale(value: str) -> float | tuple[float, float]:
    cleaned = value.strip().strip("()[]")
    if "," not in cleaned:
        return float(cleaned)
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--sample-guide-scale expects one float or two comma-separated floats")
    return float(parts[0]), float(parts[1])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        raise
