from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentRunnerConfig:
    profile_id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    frame_num: int
    fps: float
    seed: int | None
    start_image: str | None
    output_path: str
    model_root: str
    runtime_root: str
    sample_steps: int
    sample_shift: float
    sample_guide_scale: float
    offload_model: bool = True
    t5_cpu: bool = False
    dry_run: bool = False
    allow_gpu: bool = False
    lock_path: str = "renders/.render.lock"


def default_runtime_python(project_root: Path) -> Path:
    return project_root / "runtimes" / "direct-wan" / ".venv" / "Scripts" / "python.exe"
