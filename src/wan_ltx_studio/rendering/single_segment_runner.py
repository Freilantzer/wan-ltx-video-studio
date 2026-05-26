from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wan_ltx_studio.rendering.profiles import get_renderer_profile
from wan_ltx_studio.rendering.runner_config import SegmentRunnerConfig


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

    if profile.id != "wan22_ti2v_5b_fp16":
        raise SegmentRunnerError(
            f"profile {profile.id} is not executable yet; first executable path is wan22_ti2v_5b_fp16"
        )

    model_root = Path(config.model_root).resolve()
    runtime_root = Path(config.runtime_root).resolve()
    components = {component.role: model_root / component.relative_path for component in profile.components}
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
        telemetry = _run_ti2v_5b_segment(config, runtime_root, components)
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
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.textimage2video import WanTI2V
    from wan.utils.utils import save_video

    if not torch.cuda.is_available():
        raise SegmentRunnerError("CUDA is not available")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    cfg = WAN_CONFIGS["ti2v-5B"].copy()
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

    pipeline.text_encoder = _load_wan_t5_encoder(T5EncoderModel, cfg, components["text_encoder"])
    pipeline.vae = _load_wan22_vae(Wan2_2_VAE, components["vae"], pipeline.device)
    pipeline.model = _load_ti2v_model(WanModel, cfg, components["dit"])

    img = Image.open(config.start_image).convert("RGB") if config.start_image else None
    video = pipeline.generate(
        config.prompt,
        img=img,
        size=(config.width, config.height),
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

    output_path = Path(config.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=int(config.fps),
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current_allocated = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()

    del video, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "peakAllocatedBytes": peak_allocated,
        "peakReservedBytes": peak_reserved,
        "currentAllocatedBytes": current_allocated,
        "currentReservedBytes": current_reserved,
        "peakAllocatedGb": round(peak_allocated / 1024**3, 3),
        "peakReservedGb": round(peak_reserved / 1024**3, 3),
        "outputBytes": output_path.stat().st_size if output_path.exists() else None,
    }


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
    return model.eval().requires_grad_(False)


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


def _load_wan_t5_encoder(t5_cls, cfg, path: Path):
    import torch
    from safetensors.torch import load_file
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

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
    state_dict = _convert_comfy_umt5_state_dict(load_file(str(path), device="cpu"))
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise SegmentRunnerError(f"UMT5 load mismatch: missing={missing}, unexpected={unexpected}")
    encoder.model = model.to(encoder.device)
    encoder.tokenizer = HuggingfaceTokenizer(
        name=cfg.t5_tokenizer,
        seq_len=cfg.text_len,
        clean="whitespace",
    )
    return encoder


def _convert_comfy_umt5_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in state_dict.items():
        if key == "spiece_model":
            continue
        if key == "shared.weight":
            converted["token_embedding.weight"] = value
            continue
        if key == "encoder.final_layer_norm.weight":
            converted["norm.weight"] = value
            continue
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
                converted[replacements[suffix]] = value
                continue
        raise SegmentRunnerError(f"unsupported UMT5 key: {key}")
    return converted


def _normalize_wan22_vae_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    replacements = {
        "encoder.downsamples.2.downsamples.2.time_conv.weight": "encoder.downsamples.0.downsamples.2.time_conv.weight",
        "encoder.downsamples.2.downsamples.2.time_conv.bias": "encoder.downsamples.0.downsamples.2.time_conv.bias",
        "decoder.upsamples.0.upsamples.3.time_conv.weight": "decoder.upsamples.2.upsamples.3.time_conv.weight",
        "decoder.upsamples.0.upsamples.3.time_conv.bias": "decoder.upsamples.2.upsamples.3.time_conv.bias",
    }
    return {replacements.get(key, key): value for key, value in state_dict.items()}


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
    parser.add_argument("--sample-guide-scale", type=float, required=True)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--lock-path", default="renders/.render.lock")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        raise
