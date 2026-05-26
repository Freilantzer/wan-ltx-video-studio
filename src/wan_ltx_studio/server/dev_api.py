from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from wan_ltx_studio import __version__
from wan_ltx_studio.planning import (
    LoraSelection,
    PlanningError,
    SeedPolicy,
    VideoRequest,
    plan_chunked_video,
)
from wan_ltx_studio.rendering import (
    RenderingError,
    build_render_job_plan,
    list_renderer_profiles,
    render_job_plan_to_payload,
)


class StudioApiHandler(BaseHTTPRequestHandler):
    server_version = "WanLtxStudioDevApi/0.1"

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "name": "WAN/LTX Video Studio",
                    "version": __version__,
                    "engine": "direct-render-planning",
                }
            )
            return
        if self.path == "/api/render/profiles":
            self._send_json(_profiles_to_payload())
            return

        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/plan":
            self._handle_plan()
            return
        if self.path == "/api/render/plan":
            self._handle_render_plan()
            return

        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_plan(self) -> None:
        try:
            payload = self._read_json()
            request = _request_from_payload(payload)
            plan = plan_chunked_video(request)
        except (json.JSONDecodeError, TypeError, ValueError, PlanningError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(_plan_to_payload(plan))

    def _handle_render_plan(self) -> None:
        try:
            payload = self._read_json()
            request = _request_from_payload(payload)
            job = build_render_job_plan(
                request,
                model_root=_optional_str(payload, "modelRoot"),
                runtime_root=_str(payload, "runtimeRoot", "runtimes/direct-wan/src/Wan2.2"),
            )
        except (json.JSONDecodeError, TypeError, ValueError, PlanningError, RenderingError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(render_job_plan_to_payload(job))

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), StudioApiHandler)
    print(f"WAN/LTX Studio API listening on http://{host}:{port}")
    server.serve_forever()


def _request_from_payload(payload: dict[str, Any]) -> VideoRequest:
    return VideoRequest(
        width=_int(payload, "width", 1280),
        height=_int(payload, "height", 720),
        total_seconds=_float(payload, "totalSeconds", 15.0),
        fps=_float(payload, "fps", 16.0),
        chunk_seconds=_float(payload, "chunkSeconds", 5.0),
        start_image=_optional_str(payload, "startImage"),
        prompt=_str(payload, "prompt", ""),
        segment_prompts=_segment_prompts(payload),
        negative_prompt=_str(payload, "negativePrompt", ""),
        seed=_optional_int(payload, "seed"),
        seed_policy=SeedPolicy(_str(payload, "seedPolicy", SeedPolicy.FIXED.value)),
        base_model=_str(payload, "baseModel", "wan22_i2v_a14b_fp8_original"),
        loras=_loras(payload),
        pixel_budget=_int(payload, "pixelBudget", 2_100_000),
        boundary_trim_frames=_int(payload, "boundaryTrimFrames", 1),
        motion_frames=_int(payload, "motionFrames", 10),
        motion_amplitude=_float(payload, "motionAmplitude", 1.15),
    )


def _plan_to_payload(plan: Any) -> dict[str, Any]:
    return {
        "targetTimelineFrames": plan.target_timeline_frames,
        "requestedChunkFrames": plan.requested_chunk_frames,
        "actualOutputFrames": plan.actual_output_frames,
        "extraOutputFrames": plan.extra_output_frames,
        "targetDurationSeconds": plan.target_duration_seconds,
        "actualOutputDurationSeconds": plan.actual_output_duration_seconds,
        "pixels": plan.request.pixels,
        "engine": {
            "baseModel": plan.request.base_model,
            "loras": [
                {
                    "name": lora.name,
                    "strength": lora.strength,
                    "role": lora.role,
                    "enabled": lora.enabled,
                }
                for lora in plan.request.loras
                if lora.enabled
            ],
        },
        "segments": [
            {
                "index": segment.index,
                "width": segment.width,
                "height": segment.height,
                "fps": segment.fps,
                "requestedTimelineFrames": segment.requested_timeline_frames,
                "inputFrames": segment.input_frames,
                "outputFrames": segment.output_frames,
                "inputDurationSeconds": segment.input_duration_seconds,
                "outputDurationSeconds": segment.output_duration_seconds,
                "seed": segment.seed,
                "prompt": segment.prompt,
                "continuity": {
                    "source": segment.continuity.source,
                    "trimStartFrames": segment.continuity.trim_start_frames,
                    "motionFrames": segment.continuity.motion_frames,
                    "motionAmplitude": segment.continuity.motion_amplitude,
                    "previousSegmentIndex": segment.continuity.previous_segment_index,
                    "startImage": segment.continuity.start_image,
                },
            }
            for segment in plan.segments
        ],
    }


def _profiles_to_payload() -> dict[str, Any]:
    return {
        "profiles": [
            {
                "id": profile.id,
                "label": profile.label,
                "family": profile.family,
                "task": profile.task,
                "checkpointFormat": profile.checkpoint_format,
                "sampleSteps": profile.sample_steps,
                "sampleShift": profile.sample_shift,
                "sampleGuideScale": profile.sample_guide_scale,
                "fps": profile.fps,
                "vramPolicy": {
                    "targetGb": profile.vram_policy.target_gb,
                    "warnGb": profile.vram_policy.warn_gb,
                    "unsafeGb": profile.vram_policy.unsafe_gb,
                },
                "builtInLoras": [
                    {
                        "role": component.role,
                        "relativePath": component.relative_path,
                        "dtype": component.dtype,
                    }
                    for component in profile.built_in_loras
                ],
            }
            for profile in list_renderer_profiles()
        ]
    }


def _int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    return int(value)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    return float(value)


def _str(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    return str(value)


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _segment_prompts(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("segmentPrompts")
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise TypeError("segmentPrompts must be a list")
    return tuple(str(item) for item in value)


def _loras(payload: dict[str, Any]) -> tuple[LoraSelection, ...]:
    value = payload.get("loras")
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise TypeError("loras must be a list")

    loras: list[LoraSelection] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each LoRA must be an object")
        loras.append(
            LoraSelection(
                name=_str(item, "name", ""),
                strength=_float(item, "strength", 1.0),
                role=_str(item, "role", "creative"),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return tuple(loras)
