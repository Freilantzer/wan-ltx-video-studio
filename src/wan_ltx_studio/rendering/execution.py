from __future__ import annotations

from pathlib import Path

from wan_ltx_studio.rendering.jobs import RenderJobPlan
from wan_ltx_studio.rendering.runner_config import default_runtime_python


class RenderExecutionError(ValueError):
    """Raised when a planned job cannot be translated into a runner command."""


def build_single_segment_command(
    job: RenderJobPlan,
    *,
    segment_index: int = 0,
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
    dry_run: bool = True,
    allow_gpu: bool = False,
) -> list[str]:
    if job.profile.id != "wan22_ti2v_5b_fp16":
        raise RenderExecutionError(
            f"profile {job.profile.id} is not executable yet; first executable path is wan22_ti2v_5b_fp16"
        )
    if segment_index < 0 or segment_index >= len(job.commands):
        raise RenderExecutionError(f"segment index out of range: {segment_index}")

    root = Path(project_root or Path.cwd()).resolve()
    python_path = default_runtime_python(root)
    command = job.commands[segment_index]
    output = Path(output_path or root / "renders" / job.job_id / command.output_name)

    args = [
        str(python_path),
        "-m",
        "wan_ltx_studio.rendering.single_segment_runner",
        "--profile-id",
        job.profile.id,
        "--prompt",
        command.prompt,
        "--negative-prompt",
        command.negative_prompt,
        "--width",
        str(command.width),
        "--height",
        str(command.height),
        "--frame-num",
        str(command.frame_num),
        "--fps",
        str(command.fps),
        "--output",
        str(output),
        "--model-root",
        job.model_root,
        "--runtime-root",
        job.runtime_root,
        "--sample-steps",
        str(command.sample_steps),
        "--sample-shift",
        str(command.sample_shift),
        "--sample-guide-scale",
        str(command.sample_guide_scale),
        "--lock-path",
        str(root / "renders" / ".render.lock"),
    ]
    if command.seed is not None:
        args.extend(["--seed", str(command.seed)])
    if job.video_plan.request.start_image:
        args.extend(["--start-image", job.video_plan.request.start_image])
    if command.offload_model:
        args.append("--offload-model")
    else:
        args.append("--no-offload-model")
    if command.t5_cpu:
        args.append("--t5-cpu")
    if dry_run:
        args.append("--dry-run")
    if allow_gpu:
        args.append("--allow-gpu")
    return args
