import tempfile
import unittest
from pathlib import Path

from wan_ltx_studio.planning import LoraSelection, VideoRequest
from wan_ltx_studio.rendering import (
    RenderExecutionError,
    RenderingError,
    build_single_segment_command,
    build_render_job_plan,
    get_renderer_profile,
    render_job_plan_to_payload,
)
from wan_ltx_studio.rendering.single_segment_runner import (
    _CudaMemoryTelemetry,
    _crop_or_resize_video_to_request,
    _ceil_to_multiple,
    _convert_comfy_umt5_state_dict,
    _decode_wan22_vae_tiled_stream,
    _decode_wan22_vae_temporal_stream,
    _install_attention_fallback,
    _normalize_wan22_vae_state_dict,
    _parse_typeperf_gpu_memory_sample,
    _reset_wan_rope_freqs,
    _runner_warnings,
)
from wan_ltx_studio.rendering.runner_config import SegmentRunnerConfig


class RenderingBackendTests(unittest.TestCase):
    def test_lightning_profile_captures_reference_vram_policy(self):
        profile = get_renderer_profile("wan22_i2v_a14b_fp8_lightning_workflow")

        self.assertEqual(profile.task, "i2v-A14B")
        self.assertEqual(profile.sample_steps, 4)
        self.assertEqual(profile.vram_policy.target_gb, 25.0)
        self.assertEqual(profile.vram_policy.warn_gb, 28.0)
        self.assertEqual(profile.vram_policy.unsafe_gb, 30.0)
        self.assertEqual(
            [component.role for component in profile.built_in_loras],
            ["high_noise_lora", "low_noise_lora"],
        )

    def test_render_job_plan_resolves_models_and_segments_without_gpu_load(self):
        profile = get_renderer_profile("wan22_i2v_a14b_fp8_lightning_workflow")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            runtime_root = root / "runtime"
            (runtime_root / "wan").mkdir(parents=True)
            (runtime_root / "generate.py").write_text("# test runtime", encoding="utf-8")
            for component in profile.all_components:
                component_path = model_root / component.relative_path
                component_path.parent.mkdir(parents=True, exist_ok=True)
                component_path.write_bytes(b"x")

            job = build_render_job_plan(
                VideoRequest(
                    width=1280,
                    height=720,
                    fps=16,
                    total_seconds=15,
                    chunk_seconds=5,
                    start_image="inputs/start_frames_1280x720/woman_black_sand_beach.png",
                    prompt="shared city drive",
                    seed=1234,
                    base_model=profile.id,
                    loras=(LoraSelection(name="cinematic_motion", strength=0.65),),
                ),
                model_root=model_root,
                runtime_root=runtime_root,
            )
            payload = render_job_plan_to_payload(job)

        self.assertTrue(payload["executionReady"])
        self.assertTrue(payload["requiredModelFilesReady"])
        self.assertEqual(payload["vramPolicy"]["targetGb"], 25.0)
        self.assertEqual(payload["memoryStrategy"]["expertPlacement"], "move only the active high/low WAN expert to CUDA")
        self.assertEqual([command["frameNum"] for command in payload["commands"]], [81, 81, 81])
        self.assertEqual([command["outputFramesAfterTrim"] for command in payload["commands"]], [81, 80, 80])
        self.assertEqual(payload["commands"][0]["sampleSteps"], 4)
        self.assertTrue(payload["commands"][0]["offloadModel"])
        self.assertEqual(len(payload["loras"]["builtIn"]), 2)
        self.assertEqual(payload["loras"]["requested"][0]["name"], "cinematic_motion")
        self.assertEqual(
            {stage["name"]: stage["status"] for stage in payload["stages"]}["gpu_execution"],
            "ready",
        )

    def test_ti2v_profile_builds_gpu_gated_runner_command(self):
        profile = get_renderer_profile("wan22_ti2v_5b_fp16")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            runtime_root = root / "runtime"
            (runtime_root / "wan").mkdir(parents=True)
            (runtime_root / "generate.py").write_text("# test runtime", encoding="utf-8")
            for component in profile.all_components:
                component_path = model_root / component.relative_path
                component_path.parent.mkdir(parents=True, exist_ok=True)
                component_path.write_bytes(b"x")

            job = build_render_job_plan(
                VideoRequest(
                    width=1280,
                    height=720,
                    fps=24,
                    total_seconds=5,
                    chunk_seconds=5,
                    prompt="city flythrough",
                    seed=7,
                    base_model=profile.id,
                ),
                model_root=model_root,
                runtime_root=runtime_root,
            )
            command = build_single_segment_command(
                job,
                project_root=root,
                output_path=root / "renders" / "segment.mp4",
                dry_run=True,
                allow_gpu=False,
            )
            payload = render_job_plan_to_payload(job)

        self.assertTrue(payload["executionReady"])
        self.assertEqual(
            {stage["name"]: stage["status"] for stage in payload["stages"]}["gpu_execution"],
            "ready",
        )
        self.assertIn("single_segment_runner", " ".join(command))
        self.assertIn("--dry-run", command)
        self.assertNotIn("--allow-gpu", command)

    def test_runner_warns_when_calibration_frame_count_is_not_81(self):
        config = SegmentRunnerConfig(
            profile_id="wan22_ti2v_5b_fp16",
            prompt="test",
            negative_prompt="",
            width=640,
            height=352,
            frame_num=25,
            fps=24.0,
            seed=1,
            start_image=None,
            output_path="renders/test.mp4",
            model_root="models",
            runtime_root="runtimes/direct-wan/src/Wan2.2",
            sample_steps=2,
            sample_shift=5.0,
            sample_guide_scale=5.0,
        )

        self.assertIn("use 81 frames", _runner_warnings(config)[0])

        meaningful_config = SegmentRunnerConfig(
            profile_id=config.profile_id,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            frame_num=81,
            fps=config.fps,
            seed=config.seed,
            start_image=config.start_image,
            output_path=config.output_path,
            model_root=config.model_root,
            runtime_root=config.runtime_root,
            sample_steps=config.sample_steps,
            sample_shift=config.sample_shift,
            sample_guide_scale=config.sample_guide_scale,
        )
        self.assertEqual(_runner_warnings(meaningful_config), [])

    def test_cuda_memory_telemetry_records_stage_snapshots(self):
        class FakeCuda:
            def __init__(self):
                self.allocated = 1024**3
                self.reserved = 2 * 1024**3
                self.total = 32 * 1024**3

            def synchronize(self):
                return None

            def memory_allocated(self):
                return self.allocated

            def memory_reserved(self):
                return self.reserved

            def max_memory_allocated(self):
                return 3 * 1024**3

            def max_memory_reserved(self):
                return 4 * 1024**3

            def mem_get_info(self):
                return self.total - (5 * 1024**3), self.total

        class FakeTorch:
            cuda = FakeCuda()

        with tempfile.TemporaryDirectory() as temp_dir:
            telemetry_path = Path(temp_dir) / "telemetry.jsonl"
            telemetry = _CudaMemoryTelemetry(FakeTorch, telemetry_path)
            snapshot = telemetry.mark("after_load", {"component": "test"}, synchronize=True)
            summary = telemetry.summary()
            sidecar_lines = telemetry_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(snapshot["stage"], "after_load")
        self.assertEqual(snapshot["allocatedGb"], 1.0)
        self.assertEqual(snapshot["reservedGb"], 2.0)
        self.assertEqual(snapshot["driverUsedGb"], 5.0)
        self.assertEqual(snapshot["detail"]["component"], "test")
        self.assertEqual(summary["peakAllocatedGb"], 3.0)
        self.assertEqual(summary["peakReservedGb"], 4.0)
        self.assertEqual(summary["peakDriverUsedGb"], 5.0)
        self.assertEqual(summary["stages"][0]["stage"], "after_load")
        self.assertEqual(summary["telemetryPath"], str(telemetry_path.resolve()))
        self.assertEqual(len(sidecar_lines), 1)
        self.assertIn('"stage": "after_load"', sidecar_lines[0])

    def test_typeperf_gpu_memory_parser_sums_windows_dedicated_and_shared_usage(self):
        output = "\n".join(
            [
                '"(PDH-CSV 4.0)","\\\\HOST\\GPU Adapter Memory(luid_a)\\Dedicated Usage","\\\\HOST\\GPU Adapter Memory(luid_b)\\Dedicated Usage","\\\\HOST\\GPU Adapter Memory(luid_a)\\Shared Usage","\\\\HOST\\GPU Adapter Memory(luid_b)\\Shared Usage","\\\\HOST\\GPU Adapter Memory(luid_a)\\Total Committed","\\\\HOST\\GPU Adapter Memory(luid_b)\\Total Committed"',
                '"05/26/2026 19:02:23.712","1073741824.000000","2147483648.000000","536870912.000000","268435456.000000","1610612736.000000","2415919104.000000"',
                "Exiting, please wait...",
            ]
        )

        parsed = _parse_typeperf_gpu_memory_sample(output)

        self.assertEqual(parsed["windowsDedicatedUsageGb"], 3.0)
        self.assertEqual(parsed["windowsSharedUsageGb"], 0.75)
        self.assertEqual(parsed["windowsTotalCommittedGb"], 3.75)

    def test_streaming_wan_vae_decode_concatenates_temporal_chunks_on_cpu(self):
        import torch

        class FakeModel:
            z_dim = 3

            def __init__(self):
                self.clear_count = 0
                self.decoder_calls = []
                self._feat_map = [None]
                self._conv_idx = [0]

            def clear_cache(self):
                self.clear_count += 1
                self._feat_map = [None]
                self._conv_idx = [0]

            def conv2(self, z):
                return z

            def decoder(self, x, feat_cache, feat_idx, first_chunk=False):
                self.decoder_calls.append(first_chunk)
                return x

        class FakeVae:
            def __init__(self):
                self.model = FakeModel()
                self.scale = [0.0, 1.0]

        class FakeTelemetry:
            def __init__(self):
                self.stages = []

            def mark(self, stage, detail=None, synchronize=False):
                self.stages.append((stage, detail, synchronize))

        latent = torch.stack(
            [
                torch.full((3, 2, 2), 0.1),
                torch.full((3, 2, 2), 0.2),
                torch.full((3, 2, 2), 0.3),
            ],
            dim=1,
        )
        telemetry = FakeTelemetry()
        fake_vae = FakeVae()

        decoded = _decode_wan22_vae_temporal_stream(
            fake_vae,
            latent,
            unpatchify_fn=lambda tensor, patch_size: tensor,
            telemetry=telemetry,
            torch_module=torch,
        )

        self.assertEqual(tuple(decoded.shape), (3, 3, 2, 2))
        self.assertFalse(decoded.is_cuda)
        self.assertEqual(fake_vae.model.decoder_calls, [True, False, False])
        self.assertEqual([stage for stage, _, _ in telemetry.stages][0], "vae_temporal_decode_conv2_end")
        self.assertEqual([stage for stage, _, _ in telemetry.stages][-1], "vae_temporal_decode_chunk_cpu")

    def test_tiled_wan_vae_decode_accumulates_spatial_tiles_on_cpu(self):
        import torch

        class FakeModel:
            z_dim = 1

            def __init__(self):
                self.clear_count = 0
                self.decoder_calls = 0
                self._feat_map = [None]
                self._conv_idx = [0]

            def clear_cache(self):
                self.clear_count += 1
                self._feat_map = [None]
                self._conv_idx = [0]

            def conv2(self, z):
                return z

            def decoder(self, x, feat_cache, feat_idx, first_chunk=False):
                self.decoder_calls += 1
                return x

        class FakeVae:
            def __init__(self):
                self.model = FakeModel()
                self.scale = [0.0, 1.0]

        class FakeTelemetry:
            def __init__(self):
                self.stages = []

            def mark(self, stage, detail=None, synchronize=False):
                self.stages.append((stage, detail, synchronize))

        latent = torch.linspace(-0.7, 0.7, 8, dtype=torch.float32).reshape(1, 2, 2, 2)
        telemetry = FakeTelemetry()
        fake_vae = FakeVae()

        decoded = _decode_wan22_vae_tiled_stream(
            fake_vae,
            latent,
            unpatchify_fn=lambda tensor, patch_size: tensor,
            telemetry=telemetry,
            torch_module=torch,
            tile_size=(2, 1),
            tile_stride=(1, 1),
            spatial_upsample=1,
        )

        self.assertEqual(tuple(decoded.shape), (1, 2, 2, 2))
        self.assertFalse(decoded.is_cuda)
        self.assertTrue(torch.allclose(decoded, latent))
        self.assertGreater(fake_vae.model.decoder_calls, 2)
        self.assertEqual([stage for stage, _, _ in telemetry.stages][0], "vae_tiled_decode_start")
        self.assertEqual([stage for stage, _, _ in telemetry.stages][-1], "vae_tiled_decode_end")

    def test_runner_ceil_multiple_supports_wan_generation_grid(self):
        self.assertEqual(_ceil_to_multiple(720, 32), 736)
        self.assertEqual(_ceil_to_multiple(1280, 32), 1280)

    def test_video_center_crop_matches_requested_size(self):
        import torch

        video = torch.zeros((3, 81, 736, 1280), dtype=torch.float32)
        cropped, detail = _crop_or_resize_video_to_request(video, 720, 1280, torch)

        self.assertEqual(tuple(cropped.shape), (3, 81, 720, 1280))
        self.assertEqual(detail["mode"], "center_crop")
        self.assertEqual(detail["cropTop"], 8)
        self.assertEqual(detail["cropLeft"], 0)

    def test_a14b_profile_builds_gpu_gated_runner_command(self):
        profile = get_renderer_profile("wan22_i2v_a14b_fp8_lightning_workflow")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            runtime_root = root / "runtime"
            (runtime_root / "wan").mkdir(parents=True)
            (runtime_root / "generate.py").write_text("# test runtime", encoding="utf-8")
            for component in profile.all_components:
                component_path = model_root / component.relative_path
                component_path.parent.mkdir(parents=True, exist_ok=True)
                component_path.write_bytes(b"x")

            job = build_render_job_plan(
                VideoRequest(
                    width=1280,
                    height=720,
                    fps=16,
                    total_seconds=5,
                    chunk_seconds=5,
                    start_image="inputs/start_frames_1280x720/woman_black_sand_beach.png",
                    prompt="beach portrait",
                    seed=17,
                    base_model=profile.id,
                ),
                model_root=model_root,
                runtime_root=runtime_root,
            )
            command = build_single_segment_command(
                job,
                project_root=root,
                output_path=root / "renders" / "a14b.mp4",
                dry_run=True,
                allow_gpu=False,
            )
            payload = render_job_plan_to_payload(job)

        self.assertTrue(payload["executionReady"])
        self.assertIn("--sample-guide-scale", command)
        self.assertIn("1,1", command)
        self.assertIn("--dry-run", command)
        self.assertNotIn("--allow-gpu", command)

    def test_planned_only_profile_cannot_build_runner_command(self):
        job = build_render_job_plan(
            VideoRequest(
                width=1280,
                height=720,
                total_seconds=5,
                base_model="wan22_i2v_a14b_q8_gguf",
            )
        )

        with self.assertRaises(RenderExecutionError):
            build_single_segment_command(job)

    def test_missing_required_model_file_blocks_planned_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_root = root / "runtime"
            (runtime_root / "wan").mkdir(parents=True)
            (runtime_root / "generate.py").write_text("# test runtime", encoding="utf-8")

            job = build_render_job_plan(
                VideoRequest(
                    width=1280,
                    height=720,
                    total_seconds=5,
                    base_model="wan22_i2v_a14b_fp8_original",
                ),
                model_root=root / "models",
                runtime_root=runtime_root,
            )
            payload = render_job_plan_to_payload(job)

        self.assertFalse(payload["requiredModelFilesReady"])
        self.assertEqual(
            {stage["name"]: stage["status"] for stage in payload["stages"]}["resolve_model_files"],
            "blocked",
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(RenderingError):
            build_render_job_plan(
                VideoRequest(
                    width=1280,
                    height=720,
                    total_seconds=5,
                    base_model="missing-profile",
                )
            )

    def test_comfy_umt5_keys_convert_to_upstream_runner_keys(self):
        sentinel = object()
        converted = _convert_comfy_umt5_state_dict(
            {
                "shared.weight": sentinel,
                "encoder.final_layer_norm.weight": sentinel,
                "encoder.block.0.layer.0.layer_norm.weight": sentinel,
                "encoder.block.0.layer.0.SelfAttention.q.weight": sentinel,
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": sentinel,
                "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": sentinel,
                "encoder.block.0.layer.1.DenseReluDense.wi_1.weight": sentinel,
                "encoder.block.0.layer.1.DenseReluDense.wo.weight": sentinel,
                "spiece_model": sentinel,
            }
        )

        self.assertEqual(converted["token_embedding.weight"], sentinel)
        self.assertEqual(converted["norm.weight"], sentinel)
        self.assertEqual(converted["blocks.0.norm1.weight"], sentinel)
        self.assertEqual(converted["blocks.0.attn.q.weight"], sentinel)
        self.assertEqual(converted["blocks.0.pos_embedding.embedding.weight"], sentinel)
        self.assertEqual(converted["blocks.0.ffn.gate.0.weight"], sentinel)
        self.assertEqual(converted["blocks.0.ffn.fc1.weight"], sentinel)
        self.assertEqual(converted["blocks.0.ffn.fc2.weight"], sentinel)
        self.assertNotIn("spiece_model", converted)

    def test_wan22_vae_keys_are_kept_for_official_class_layout(self):
        sentinel = object()
        state_dict = {
            "encoder.downsamples.2.downsamples.2.time_conv.weight": sentinel,
            "decoder.upsamples.0.upsamples.3.time_conv.weight": sentinel,
        }

        self.assertEqual(_normalize_wan22_vae_state_dict(state_dict), state_dict)

    def test_rope_freq_reset_replaces_meta_placeholder(self):
        class Config:
            dim = 3072
            num_heads = 24

        class Model:
            freqs = None

        model = Model()
        _reset_wan_rope_freqs(model, Config())

        self.assertEqual(tuple(model.freqs.shape), (1024, 64))

    def test_attention_fallback_installs_without_flash_attention(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path("runtimes/direct-wan/src/Wan2.2").resolve()))
        try:
            import wan.modules.attention as attention_module
            import wan.modules.model as model_module
        except ModuleNotFoundError as exc:
            self.skipTest(f"direct WAN runtime dependency unavailable: {exc}")

        if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE:
            self.skipTest("flash attention is installed")

        _install_attention_fallback()

        self.assertIs(attention_module.flash_attention, model_module.flash_attention)

        import torch

        q = torch.randn(1, 4, 2, 3)
        k = torch.randn(1, 4, 2, 3)
        v = torch.randn(1, 4, 2, 3)
        out = attention_module.flash_attention(
            q,
            k,
            v,
            k_lens=torch.tensor([2]),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(out.shape), (1, 4, 2, 3))
        self.assertTrue(torch.isfinite(out).all())

        trimmed_out = attention_module.flash_attention(
            q,
            k,
            v,
            q_lens=torch.tensor([3]),
            k_lens=torch.tensor([2]),
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(trimmed_out[:, 3:], torch.zeros_like(trimmed_out[:, 3:])))


if __name__ == "__main__":
    unittest.main()
