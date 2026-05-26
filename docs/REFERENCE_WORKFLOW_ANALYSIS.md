# Reference Workflow Analysis

Date: 2026-05-26

Source files inspected locally:

```text
C:\Users\freil\Downloads\wan2-2 workflow.json
C:\Users\freil\Downloads\Wan2.2_00020.mp4
```

The raw workflow is not committed because it contains private model/library references. This note records the reusable architecture only.

## Output

`Wan2.2_00020.mp4`:

- 1280x720
- 16 fps
- 241 frames
- 15.0625 seconds
- H.264 MP4, yuv420p
- Size: about 8.56 MB

User-observed run:

- 3 x about 5 second chunks
- Around 321 seconds total
- Around 80% dedicated VRAM in the user's working Comfy setup
- GPU saturated near 99-100%

## Workflow Shape

The workflow is a three-segment WAN 2.2 image-to-video chain:

```text
Segment 1:
  start image -> PainterLongVideo -> high-noise sampler -> low-noise sampler -> VAE decode

Segment 2:
  segment 1 decoded frames -> PainterLongVideo -> high-noise sampler -> low-noise sampler -> VAE decode
  trim duplicate first frame

Segment 3:
  segment 2 decoded frames -> PainterLongVideo -> high-noise sampler -> low-noise sampler -> VAE decode
  trim duplicate first frame

Final:
  merge segment frames -> H.264 MP4
```

Frame math:

```text
Segment 1: 81 frames
Segment 2: 80 frames after dropping duplicate first frame
Segment 3: 80 frames after dropping duplicate first frame
Total: 241 frames at 16 fps = 15.0625 seconds
```

## Important Nodes

- `PainterLongVideo`: continuity node for WAN 2.2 long-video chaining.
- `KSamplerAdvanced`: two samplers per segment.
- `UNETLoader`: high-noise and low-noise WAN models.
- `PathchSageAttentionKJ`: per-model SageAttention patch.
- `ModelSamplingSD3`: shift 5.0.
- `GetImageRangeFromBatch`: trims duplicate first frames from later segments.
- `VHS_MergeImages`: concatenates decoded frame batches.
- `VHS_VideoCombine`: writes final MP4.

## Key Settings

- Width: 1280
- Height: 720
- Frames per segment: 81
- Motion frames: 10
- Motion amplitude: 1.15
- Sampler: euler
- Scheduler: simple
- CFG: 1.0
- Steps: 4 total split across high/low phases
- High phase: start step 0, end step 2, return leftover noise
- Low phase: start step 2, finish remaining steps, no leftover noise

The LoRA loader nodes in the inspected workflow are bypassed. The app should not assume LoRA nodes are active just because they are present in a graph.

## App Implications

This is the better architectural reference than a one-shot monolithic Comfy test:

- Treat video generation as chunked timeline rendering.
- Make chunk length explicit, for example 81 frames at 16 fps.
- Carry previous decoded frames into the next chunk.
- Use the last frame as the continuity start point.
- Use optional first-frame/global reference anchoring to reduce drift.
- Trim duplicate boundary frames before concatenation.
- Concat segments automatically into one deliverable.
- Track memory per chunk, not only per full job.
- Patch attention per model/workflow where useful, instead of relying only on global runtime flags.

## Product Translation

The app should expose this as simple controls:

- Total duration
- Chunk duration
- Resolution
- FPS
- Start image
- Prompt per whole video or per segment
- Continuity strength
- Motion frames
- Motion amplitude
- Seed behavior
- Memory profile

Internally, the engine adapter can generate a workflow graph equivalent to:

```text
VideoJob
  -> SegmentPlan[]
  -> Render segment
  -> Extract boundary frame/reference frames
  -> Render next segment
  -> Drop duplicate boundary frames
  -> Concat
  -> Save final video plus metadata
```

## What This Changes

The earlier A14B `--highvram` test remains useful only as a pressure test. It was not representative of the user's working setup.

Next implementation should focus on a typed chunked-generation adapter based on this reference workflow, not more ad hoc Comfy benchmarking.
