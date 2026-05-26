from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil

DEFAULT_RTX_5090_PIXEL_BUDGET = 2_100_000
DEFAULT_DIMENSION_MULTIPLE = 16
WAN_FRAME_STRIDE = 4
FRAME_EPSILON = 1e-9


class PlanningError(ValueError):
    """Raised when a video request cannot be represented safely."""


class SeedPolicy(str, Enum):
    FIXED = "fixed"
    INCREMENT = "increment"
    NONE = "none"


@dataclass(frozen=True)
class LoraSelection:
    name: str
    strength: float = 1.0
    role: str = "creative"
    enabled: bool = True


@dataclass(frozen=True)
class VideoRequest:
    width: int
    height: int
    total_seconds: float
    fps: float = 16.0
    chunk_seconds: float = 5.0
    start_image: str | None = None
    prompt: str = ""
    segment_prompts: tuple[str, ...] = ()
    negative_prompt: str = ""
    seed: int | None = None
    seed_policy: SeedPolicy | str = SeedPolicy.FIXED
    base_model: str = "wan22_i2v_a14b_fp8_original"
    loras: tuple[LoraSelection, ...] = ()
    pixel_budget: int = DEFAULT_RTX_5090_PIXEL_BUDGET
    dimension_multiple: int = DEFAULT_DIMENSION_MULTIPLE
    frame_stride: int = WAN_FRAME_STRIDE
    boundary_trim_frames: int = 1
    motion_frames: int = 10
    motion_amplitude: float = 1.15

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class ContinuityPlan:
    source: str
    trim_start_frames: int
    motion_frames: int
    motion_amplitude: float
    previous_segment_index: int | None = None
    start_image: str | None = None


@dataclass(frozen=True)
class SegmentPlan:
    index: int
    width: int
    height: int
    fps: float
    requested_timeline_frames: int
    input_frames: int
    output_frames: int
    seed: int | None
    prompt: str
    negative_prompt: str
    continuity: ContinuityPlan

    @property
    def input_duration_seconds(self) -> float:
        return self.input_frames / self.fps

    @property
    def output_duration_seconds(self) -> float:
        return self.output_frames / self.fps


@dataclass(frozen=True)
class ChunkedVideoPlan:
    request: VideoRequest
    target_timeline_frames: int
    requested_chunk_frames: int
    segments: tuple[SegmentPlan, ...]

    @property
    def actual_output_frames(self) -> int:
        return sum(segment.output_frames for segment in self.segments)

    @property
    def extra_output_frames(self) -> int:
        return self.actual_output_frames - self.target_timeline_frames

    @property
    def target_duration_seconds(self) -> float:
        return self.target_timeline_frames / self.request.fps

    @property
    def actual_output_duration_seconds(self) -> float:
        return self.actual_output_frames / self.request.fps


def next_wan_frame_count(min_frames: int, stride: int = WAN_FRAME_STRIDE) -> int:
    """Return the next frame count compatible with WAN's 4n+1 latent cadence."""
    if stride < 1:
        raise PlanningError("frame stride must be at least 1")
    if min_frames < 1:
        raise PlanningError("minimum frame count must be at least 1")

    remainder = (min_frames - 1) % stride
    if remainder == 0:
        return min_frames
    return min_frames + (stride - remainder)


def plan_chunked_video(request: VideoRequest) -> ChunkedVideoPlan:
    _validate_request(request)

    target_timeline_frames = _ceil_frames(request.total_seconds, request.fps)
    requested_chunk_frames = _ceil_frames(request.chunk_seconds, request.fps)
    seed_policy = SeedPolicy(request.seed_policy)

    remaining_frames = target_timeline_frames
    segments: list[SegmentPlan] = []

    while remaining_frames > 0:
        index = len(segments)
        requested_frames = min(remaining_frames, requested_chunk_frames)
        trim_start_frames = 0 if index == 0 else request.boundary_trim_frames
        input_frames = next_wan_frame_count(
            requested_frames + trim_start_frames,
            request.frame_stride,
        )
        output_frames = input_frames - trim_start_frames

        if output_frames <= 0:
            raise PlanningError("segment output frames must be positive")

        segments.append(
            SegmentPlan(
                index=index,
                width=request.width,
                height=request.height,
                fps=request.fps,
                requested_timeline_frames=requested_frames,
                input_frames=input_frames,
                output_frames=output_frames,
                seed=_segment_seed(request.seed, seed_policy, index),
                prompt=_segment_prompt(request, index),
                negative_prompt=request.negative_prompt,
                continuity=ContinuityPlan(
                    source=_continuity_source(request, index),
                    previous_segment_index=None if index == 0 else index - 1,
                    start_image=request.start_image if index == 0 else None,
                    trim_start_frames=trim_start_frames,
                    motion_frames=request.motion_frames,
                    motion_amplitude=request.motion_amplitude,
                ),
            )
        )
        remaining_frames -= requested_frames

    return ChunkedVideoPlan(
        request=request,
        target_timeline_frames=target_timeline_frames,
        requested_chunk_frames=requested_chunk_frames,
        segments=tuple(segments),
    )


def _validate_request(request: VideoRequest) -> None:
    if request.width <= 0 or request.height <= 0:
        raise PlanningError("width and height must be positive")
    if request.total_seconds <= 0:
        raise PlanningError("total_seconds must be positive")
    if request.fps <= 0:
        raise PlanningError("fps must be positive")
    if request.chunk_seconds <= 0:
        raise PlanningError("chunk_seconds must be positive")
    if request.pixel_budget <= 0:
        raise PlanningError("pixel_budget must be positive")
    if request.pixels > request.pixel_budget:
        raise PlanningError(
            f"resolution uses {request.pixels} pixels, above budget {request.pixel_budget}"
        )
    if request.dimension_multiple <= 0:
        raise PlanningError("dimension_multiple must be positive")
    if request.width % request.dimension_multiple != 0:
        raise PlanningError(
            f"width must be divisible by {request.dimension_multiple}"
        )
    if request.height % request.dimension_multiple != 0:
        raise PlanningError(
            f"height must be divisible by {request.dimension_multiple}"
        )
    if request.frame_stride <= 0:
        raise PlanningError("frame_stride must be positive")
    if request.boundary_trim_frames < 0:
        raise PlanningError("boundary_trim_frames cannot be negative")
    if request.motion_frames < 0:
        raise PlanningError("motion_frames cannot be negative")
    if request.motion_amplitude < 1.0:
        raise PlanningError("motion_amplitude must be at least 1.0")
    if not request.base_model:
        raise PlanningError("base_model is required")
    for lora in request.loras:
        if not lora.name:
            raise PlanningError("LoRA name is required")
    SeedPolicy(request.seed_policy)


def _ceil_frames(seconds: float, fps: float) -> int:
    return max(1, int(ceil((seconds * fps) - FRAME_EPSILON)))


def _segment_seed(seed: int | None, policy: SeedPolicy, segment_index: int) -> int | None:
    if seed is None or policy == SeedPolicy.NONE:
        return None
    if policy == SeedPolicy.FIXED:
        return seed
    if policy == SeedPolicy.INCREMENT:
        return seed + segment_index
    raise PlanningError(f"unsupported seed policy: {policy}")


def _continuity_source(request: VideoRequest, segment_index: int) -> str:
    if segment_index > 0:
        return "previous_segment"
    if request.start_image:
        return "start_image"
    return "none"


def _segment_prompt(request: VideoRequest, segment_index: int) -> str:
    if segment_index < len(request.segment_prompts):
        prompt = request.segment_prompts[segment_index].strip()
        if prompt:
            return prompt
    return request.prompt
