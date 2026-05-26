from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from wan_ltx_studio.planning import (
    ChunkedVideoPlan,
    LoraSelection,
    VideoRequest,
    plan_chunked_video,
)
from wan_ltx_studio.rendering.profiles import (
    RendererProfile,
    RenderingProfileError,
    ResolvedModelComponent,
    get_renderer_profile,
)


class RenderingError(ValueError):
    """Raised when a render job cannot be planned."""


@dataclass(frozen=True)
class RenderStage:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class SegmentRenderCommand:
    segment_index: int
    task: str
    width: int
    height: int
    frame_num: int
    fps: float
    prompt: str
    negative_prompt: str
    seed: int | None
    start_source: str
    output_frames_after_trim: int
    trim_start_frames: int
    sample_steps: int
    sample_shift: float
    sample_guide_scale: tuple[float, float] | float
    offload_model: bool
    t5_cpu: bool
    output_name: str


@dataclass(frozen=True)
class RenderJobPlan:
    job_id: str
    created_at: str
    renderer: str
    status: str
    execution_ready: bool
    blocked_reason: str
    profile: RendererProfile
    video_plan: ChunkedVideoPlan
    model_root: str
    runtime_root: str
    resolved_components: tuple[ResolvedModelComponent, ...]
    requested_loras: tuple[LoraSelection, ...]
    stages: tuple[RenderStage, ...]
    commands: tuple[SegmentRenderCommand, ...]
    environment: dict[str, str]

    @property
    def required_model_files_ready(self) -> bool:
        return all(component.exists for component in self.resolved_components if component.required)


DEFAULT_ALLOCATOR_CONFIG = "max_split_size_mb:256,expandable_segments:True"
DEFAULT_RUNTIME_ROOT = Path("runtimes/direct-wan/src/Wan2.2")


def build_render_job_plan(
    request: VideoRequest,
    *,
    model_root: str | Path | None = None,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
) -> RenderJobPlan:
    try:
        profile = get_renderer_profile(request.base_model)
    except RenderingProfileError as exc:
        raise RenderingError(str(exc)) from exc

    selected_model_root = Path(
        model_root or os.environ.get("WAN_LTX_MODEL_ROOT") or "models"
    ).resolve()
    selected_runtime_root = Path(runtime_root).resolve()

    video_plan = plan_chunked_video(request)
    resolved_components = profile.resolve_components(selected_model_root)
    commands = tuple(_command_for_segment(profile, segment) for segment in video_plan.segments)
    stages = _build_stages(profile, resolved_components, selected_runtime_root, request)

    execution_ready = False
    blocked_reason = "Direct WAN GPU execution is not enabled yet; this endpoint builds the measured run shape only."

    return RenderJobPlan(
        job_id=f"render-{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc).isoformat(),
        renderer="direct-wan",
        status="planned",
        execution_ready=execution_ready,
        blocked_reason=blocked_reason,
        profile=profile,
        video_plan=video_plan,
        model_root=str(selected_model_root),
        runtime_root=str(selected_runtime_root),
        resolved_components=resolved_components,
        requested_loras=tuple(lora for lora in request.loras if lora.enabled),
        stages=stages,
        commands=commands,
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": DEFAULT_ALLOCATOR_CONFIG,
            "WAN_LTX_MEMORY_POLICY": "single_active_wan_expert",
        },
    )


def render_job_plan_to_payload(job: RenderJobPlan) -> dict:
    return {
        "jobId": job.job_id,
        "createdAt": job.created_at,
        "renderer": job.renderer,
        "status": job.status,
        "executionReady": job.execution_ready,
        "blockedReason": job.blocked_reason,
        "modelRoot": job.model_root,
        "runtimeRoot": job.runtime_root,
        "requiredModelFilesReady": job.required_model_files_ready,
        "profile": {
            "id": job.profile.id,
            "label": job.profile.label,
            "family": job.profile.family,
            "task": job.profile.task,
            "checkpointFormat": job.profile.checkpoint_format,
            "sampleSteps": job.profile.sample_steps,
            "sampleShift": job.profile.sample_shift,
            "sampleGuideScale": job.profile.sample_guide_scale,
            "fps": job.profile.fps,
            "notes": list(job.profile.notes),
        },
        "vramPolicy": {
            "targetGb": job.profile.vram_policy.target_gb,
            "warnGb": job.profile.vram_policy.warn_gb,
            "unsafeGb": job.profile.vram_policy.unsafe_gb,
        },
        "memoryStrategy": {
            "allocator": job.environment["PYTORCH_CUDA_ALLOC_CONF"],
            "expertPlacement": "move only the active high/low WAN expert to CUDA",
            "textEncoder": "encode, then offload before diffusion when offload is active",
            "vae": "decode after diffusion with DiT experts returned to CPU",
            "processPolicy": "one GPU render job at a time",
        },
        "components": [
            {
                "role": component.role,
                "relativePath": component.relative_path,
                "absolutePath": component.absolute_path,
                "dtype": component.dtype,
                "format": component.format,
                "required": component.required,
                "exists": component.exists,
                "sizeBytes": component.size_bytes,
            }
            for component in job.resolved_components
        ],
        "loras": {
            "builtIn": [
                {
                    "role": component.role,
                    "relativePath": component.relative_path,
                    "dtype": component.dtype,
                    "format": component.format,
                }
                for component in job.profile.built_in_loras
            ],
            "requested": [
                {
                    "name": lora.name,
                    "role": lora.role,
                    "strength": lora.strength,
                    "enabled": lora.enabled,
                }
                for lora in job.requested_loras
            ],
        },
        "stages": [
            {"name": stage.name, "status": stage.status, "detail": stage.detail}
            for stage in job.stages
        ],
        "commands": [
            {
                "segmentIndex": command.segment_index,
                "task": command.task,
                "width": command.width,
                "height": command.height,
                "frameNum": command.frame_num,
                "fps": command.fps,
                "prompt": command.prompt,
                "negativePrompt": command.negative_prompt,
                "seed": command.seed,
                "startSource": command.start_source,
                "outputFramesAfterTrim": command.output_frames_after_trim,
                "trimStartFrames": command.trim_start_frames,
                "sampleSteps": command.sample_steps,
                "sampleShift": command.sample_shift,
                "sampleGuideScale": command.sample_guide_scale,
                "offloadModel": command.offload_model,
                "t5Cpu": command.t5_cpu,
                "outputName": command.output_name,
            }
            for command in job.commands
        ],
    }


def _command_for_segment(profile: RendererProfile, segment) -> SegmentRenderCommand:
    return SegmentRenderCommand(
        segment_index=segment.index,
        task=profile.task,
        width=segment.width,
        height=segment.height,
        frame_num=segment.input_frames,
        fps=segment.fps,
        prompt=segment.prompt,
        negative_prompt=segment.negative_prompt,
        seed=segment.seed,
        start_source=segment.continuity.source,
        output_frames_after_trim=segment.output_frames,
        trim_start_frames=segment.continuity.trim_start_frames,
        sample_steps=profile.sample_steps,
        sample_shift=profile.sample_shift,
        sample_guide_scale=profile.sample_guide_scale,
        offload_model=True,
        t5_cpu=False,
        output_name=f"segment_{segment.index + 1:03d}.mp4",
    )


def _build_stages(
    profile: RendererProfile,
    resolved_components: tuple[ResolvedModelComponent, ...],
    runtime_root: Path,
    request: VideoRequest,
) -> tuple[RenderStage, ...]:
    missing_required = [
        component.relative_path
        for component in resolved_components
        if component.required and not component.exists
    ]
    runtime_ready = (runtime_root / "generate.py").is_file() and (runtime_root / "wan").is_dir()
    fps_status = "complete" if request.fps == profile.fps else "warning"
    fps_detail = (
        f"request fps matches profile fps ({profile.fps:g})"
        if request.fps == profile.fps
        else f"request fps {request.fps:g} differs from profile baseline {profile.fps:g}"
    )

    return (
        RenderStage(
            name="validate_request",
            status="complete",
            detail=f"{request.width}x{request.height}, {request.total_seconds:g}s, {request.fps:g} fps",
        ),
        RenderStage(
            name="resolve_model_files",
            status="complete" if not missing_required else "blocked",
            detail="all required model files found"
            if not missing_required
            else f"missing required model files: {', '.join(missing_required)}",
        ),
        RenderStage(
            name="check_direct_wan_runtime",
            status="complete" if runtime_ready else "blocked",
            detail="direct WAN runtime is present"
            if runtime_ready
            else f"direct WAN runtime not found at {runtime_root}",
        ),
        RenderStage(
            name="match_reference_fps",
            status=fps_status,
            detail=fps_detail,
        ),
        RenderStage(
            name="apply_memory_policy",
            status="complete",
            detail="allocator set; high/low experts planned as mutually exclusive CUDA residents",
        ),
        RenderStage(
            name="gpu_execution",
            status="pending",
            detail="not wired yet; next slice connects the direct WAN runner behind this plan",
        ),
    )
