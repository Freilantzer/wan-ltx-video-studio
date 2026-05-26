from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RenderingProfileError(ValueError):
    """Raised when a requested renderer profile is unknown or invalid."""


@dataclass(frozen=True)
class VramPolicy:
    target_gb: float
    warn_gb: float
    unsafe_gb: float


@dataclass(frozen=True)
class ModelComponent:
    role: str
    relative_path: str
    dtype: str
    format: str
    required: bool = True


@dataclass(frozen=True)
class ResolvedModelComponent:
    role: str
    relative_path: str
    absolute_path: str
    dtype: str
    format: str
    required: bool
    exists: bool
    size_bytes: int | None


@dataclass(frozen=True)
class RendererProfile:
    id: str
    label: str
    family: str
    task: str
    checkpoint_format: str
    sample_steps: int
    sample_shift: float
    sample_guide_scale: tuple[float, float] | float
    fps: float
    vram_policy: VramPolicy
    components: tuple[ModelComponent, ...]
    built_in_loras: tuple[ModelComponent, ...] = ()
    notes: tuple[str, ...] = ()

    def resolve_components(self, model_root: Path) -> tuple[ResolvedModelComponent, ...]:
        return tuple(_resolve_component(model_root, component) for component in self.all_components)

    @property
    def all_components(self) -> tuple[ModelComponent, ...]:
        return self.components + self.built_in_loras


WAN_A14B_720P_VRAM_POLICY = VramPolicy(target_gb=25.0, warn_gb=28.0, unsafe_gb=30.0)


_WAN_I2V_A14B_COMPONENTS = (
    ModelComponent(
        role="high_noise_dit",
        relative_path="diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        dtype="fp8_scaled",
        format="safetensors",
    ),
    ModelComponent(
        role="low_noise_dit",
        relative_path="diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        dtype="fp8_scaled",
        format="safetensors",
    ),
    ModelComponent(
        role="text_encoder",
        relative_path="text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        dtype="fp8_scaled",
        format="safetensors",
    ),
    ModelComponent(
        role="vae",
        relative_path="vae/wan_2.1_vae.safetensors",
        dtype="fp16",
        format="safetensors",
    ),
)

_WAN_I2V_LIGHTNING_LORAS = (
    ModelComponent(
        role="high_noise_lora",
        relative_path="loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
        dtype="fp16",
        format="safetensors",
    ),
    ModelComponent(
        role="low_noise_lora",
        relative_path="loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
        dtype="fp16",
        format="safetensors",
    ),
)

_PROFILES: dict[str, RendererProfile] = {
    "wan22_i2v_a14b_fp8_original": RendererProfile(
        id="wan22_i2v_a14b_fp8_original",
        label="WAN 2.2 I2V A14B FP8",
        family="WAN 2.2",
        task="i2v-A14B",
        checkpoint_format="comfy_safetensors",
        sample_steps=40,
        sample_shift=5.0,
        sample_guide_scale=(3.5, 3.5),
        fps=16.0,
        vram_policy=WAN_A14B_720P_VRAM_POLICY,
        components=_WAN_I2V_A14B_COMPONENTS,
        notes=(
            "Direct renderer must keep high/low experts mutually exclusive on CUDA.",
            "Text encoder may visit CUDA for encoding, then return to system RAM.",
        ),
    ),
    "wan22_i2v_a14b_fp8_lightning_workflow": RendererProfile(
        id="wan22_i2v_a14b_fp8_lightning_workflow",
        label="WAN 2.2 I2V A14B FP8 + Lightning",
        family="WAN 2.2",
        task="i2v-A14B",
        checkpoint_format="comfy_safetensors",
        sample_steps=4,
        sample_shift=5.0,
        sample_guide_scale=(1.0, 1.0),
        fps=16.0,
        vram_policy=WAN_A14B_720P_VRAM_POLICY,
        components=_WAN_I2V_A14B_COMPONENTS,
        built_in_loras=_WAN_I2V_LIGHTNING_LORAS,
        notes=(
            "Matches the known 720p reference class: 81-frame chunks and 4-step Lightning.",
            "Lightning LoRAs are treated as workflow components instead of optional creative LoRAs.",
        ),
    ),
    "wan22_ti2v_5b_fp16": RendererProfile(
        id="wan22_ti2v_5b_fp16",
        label="WAN 2.2 TI2V 5B",
        family="WAN 2.2",
        task="ti2v-5B",
        checkpoint_format="comfy_safetensors",
        sample_steps=50,
        sample_shift=5.0,
        sample_guide_scale=5.0,
        fps=24.0,
        vram_policy=VramPolicy(target_gb=16.0, warn_gb=22.0, unsafe_gb=28.0),
        components=(
            ModelComponent(
                role="dit",
                relative_path="diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
                dtype="fp16",
                format="safetensors",
            ),
            ModelComponent(
                role="text_encoder",
                relative_path="text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                dtype="fp8_scaled",
                format="safetensors",
            ),
            ModelComponent(
                role="vae",
                relative_path="vae/wan2.2_vae.safetensors",
                dtype="fp16",
                format="safetensors",
            ),
        ),
    ),
    "wan22_i2v_a14b_q8_gguf": RendererProfile(
        id="wan22_i2v_a14b_q8_gguf",
        label="WAN 2.2 I2V A14B Q8 GGUF",
        family="WAN 2.2",
        task="i2v-A14B",
        checkpoint_format="gguf",
        sample_steps=40,
        sample_shift=5.0,
        sample_guide_scale=(3.5, 3.5),
        fps=16.0,
        vram_policy=VramPolicy(target_gb=22.0, warn_gb=27.0, unsafe_gb=30.0),
        components=(
            ModelComponent(
                role="high_noise_dit",
                relative_path="unet/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
                dtype="q8_0",
                format="gguf",
            ),
            ModelComponent(
                role="low_noise_dit",
                relative_path="unet/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
                dtype="q8_0",
                format="gguf",
            ),
            ModelComponent(
                role="text_encoder",
                relative_path="text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                dtype="fp8_scaled",
                format="safetensors",
            ),
            ModelComponent(
                role="vae",
                relative_path="vae/wan_2.1_vae.safetensors",
                dtype="fp16",
                format="safetensors",
            ),
        ),
    ),
    "ltx23_dev_distilled": RendererProfile(
        id="ltx23_dev_distilled",
        label="LTX 2.3 Dev Distilled",
        family="LTX",
        task="ltx-i2v",
        checkpoint_format="comfy_safetensors",
        sample_steps=8,
        sample_shift=1.0,
        sample_guide_scale=1.0,
        fps=24.0,
        vram_policy=VramPolicy(target_gb=24.0, warn_gb=28.0, unsafe_gb=30.0),
        components=(
            ModelComponent(
                role="transformer",
                relative_path="diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v2.safetensors",
                dtype="fp8_scaled",
                format="safetensors",
            ),
            ModelComponent(
                role="text_projection",
                relative_path="text_encoders/ltx-2.3_text_projection_bf16.safetensors",
                dtype="bf16",
                format="safetensors",
            ),
            ModelComponent(
                role="vae",
                relative_path="vae/LTX23_video_vae_bf16.safetensors",
                dtype="bf16",
                format="safetensors",
            ),
        ),
        built_in_loras=(
            ModelComponent(
                role="distilled_lora",
                relative_path="loras/ltx-2.3-22b-distilled-lora-384.safetensors",
                dtype="fp16",
                format="safetensors",
                required=False,
            ),
        ),
    ),
}


def get_renderer_profile(profile_id: str) -> RendererProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise RenderingProfileError(f"unknown renderer profile: {profile_id}") from exc


def list_renderer_profiles() -> tuple[RendererProfile, ...]:
    return tuple(_PROFILES.values())


def _resolve_component(model_root: Path, component: ModelComponent) -> ResolvedModelComponent:
    absolute_path = (model_root / component.relative_path).resolve()
    exists = absolute_path.is_file()
    return ResolvedModelComponent(
        role=component.role,
        relative_path=component.relative_path,
        absolute_path=str(absolute_path),
        dtype=component.dtype,
        format=component.format,
        required=component.required,
        exists=exists,
        size_bytes=absolute_path.stat().st_size if exists else None,
    )
