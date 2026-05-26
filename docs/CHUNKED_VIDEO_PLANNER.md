# Chunked Video Planner

Date: 2026-05-26

The first app-owned runtime primitive is a chunked video planner. It converts simple product settings into exact segment jobs before any ComfyUI or direct model backend is involved.

## Why This Exists

The user's proven WAN 2.2 workflow generated 3 x 5 second chunks at 1280 x 720 and 16 fps, then stitched them into a 241 frame video. The extra frame is expected: WAN-compatible video lengths follow a 4n+1 cadence, so a 5 second / 16 fps chunk uses 81 frames, and later chunks trim their duplicate first frame.

This planner captures that behavior as typed data:

- `VideoRequest`: user-facing shot settings.
- `SegmentPlan`: one backend job.
- `ContinuityPlan`: how each segment starts, trims, and references the previous segment.
- `ChunkedVideoPlan`: final derived timeline, actual output duration, and frame counts.

## Reference Example

For 15 seconds at 16 fps with 5 second chunks:

```text
requested timeline frames: 240
segment input frames:      81, 81, 81
segment output frames:     81, 80, 80
trim start frames:         0, 1, 1
final output frames:       241
final output duration:     15.0625 seconds
```

## Validation

The planner currently validates:

- positive width, height, duration, fps, and chunk duration
- pixel budget, defaulting to an RTX 5090-oriented 2.1 MP budget
- dimensions divisible by 16
- WAN 4n+1 frame cadence
- non-negative boundary trim and motion settings

The default pixel budget allows 1280 x 1600, 1920 x 1080, and equivalent 32 GB VRAM test targets while rejecting larger accidental jumps until a profile explicitly opts in.

## Next Adapter Step

The Comfy adapter should consume `SegmentPlan` objects and generate an API workflow using the native WAN/PainterLongVideo path:

- first segment receives a start image or text-only start
- later segments receive previous segment frames/video
- KSampler high and low phases use the chosen model profile
- duplicate boundary frames are trimmed before concat
- output metadata records every segment plan

`ComfyUI-WanVideoWrapper` should be evaluated after this baseline for block swap, offload, cache, quantization, and control workflows.
