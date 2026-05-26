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
from wan_ltx_studio.rendering.single_segment_runner import _convert_comfy_umt5_state_dict


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

        self.assertFalse(payload["executionReady"])
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
            "pending",
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

    def test_non_executable_profile_cannot_build_runner_command(self):
        job = build_render_job_plan(
            VideoRequest(
                width=1280,
                height=720,
                total_seconds=5,
                base_model="wan22_i2v_a14b_fp8_original",
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


if __name__ == "__main__":
    unittest.main()
