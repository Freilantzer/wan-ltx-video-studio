import unittest

from wan_ltx_studio.planning import (
    PlanningError,
    SeedPolicy,
    VideoRequest,
    next_wan_frame_count,
    plan_chunked_video,
)


class ChunkedVideoPlannerTests(unittest.TestCase):
    def test_wan_frame_count_uses_4n_plus_1_cadence(self):
        self.assertEqual(next_wan_frame_count(1), 1)
        self.assertEqual(next_wan_frame_count(80), 81)
        self.assertEqual(next_wan_frame_count(81), 81)
        self.assertEqual(next_wan_frame_count(82), 85)

    def test_reference_three_by_five_second_plan(self):
        plan = plan_chunked_video(
            VideoRequest(
                width=1280,
                height=720,
                fps=16,
                total_seconds=15,
                chunk_seconds=5,
                start_image="inputs/start_frames_1280x720/woman_black_sand_beach.png",
                seed=1234,
            )
        )

        self.assertEqual(len(plan.segments), 3)
        self.assertEqual(plan.target_timeline_frames, 240)
        self.assertEqual([segment.input_frames for segment in plan.segments], [81, 81, 81])
        self.assertEqual([segment.output_frames for segment in plan.segments], [81, 80, 80])
        self.assertEqual([segment.continuity.trim_start_frames for segment in plan.segments], [0, 1, 1])
        self.assertEqual(plan.actual_output_frames, 241)
        self.assertAlmostEqual(plan.actual_output_duration_seconds, 15.0625)
        self.assertEqual(plan.segments[0].continuity.source, "start_image")
        self.assertEqual(plan.segments[1].continuity.source, "previous_segment")
        self.assertEqual(plan.segments[1].continuity.previous_segment_index, 0)

    def test_frame_rounding_tolerates_float_noise(self):
        plan = plan_chunked_video(
            VideoRequest(width=1280, height=720, fps=10, total_seconds=0.1 + 0.2, chunk_seconds=1)
        )

        self.assertEqual(plan.target_timeline_frames, 3)

    def test_partial_final_segment_is_shortened_but_still_wan_compatible(self):
        plan = plan_chunked_video(
            VideoRequest(width=1280, height=720, fps=16, total_seconds=12, chunk_seconds=5)
        )

        self.assertEqual([segment.requested_timeline_frames for segment in plan.segments], [80, 80, 32])
        self.assertEqual([segment.input_frames for segment in plan.segments], [81, 81, 33])
        self.assertEqual([segment.output_frames for segment in plan.segments], [81, 80, 32])
        self.assertEqual(plan.actual_output_frames, 193)

    def test_portrait_1280_by_1600_fits_default_rtx_5090_budget(self):
        plan = plan_chunked_video(
            VideoRequest(width=1280, height=1600, fps=16, total_seconds=5, chunk_seconds=5)
        )

        self.assertEqual(plan.request.pixels, 2_048_000)
        self.assertEqual(len(plan.segments), 1)

    def test_pixel_budget_rejects_overlarge_resolution(self):
        with self.assertRaises(PlanningError):
            plan_chunked_video(
                VideoRequest(width=1536, height=1536, fps=16, total_seconds=5, chunk_seconds=5)
            )

    def test_dimensions_must_be_divisible_by_multiple(self):
        with self.assertRaises(PlanningError):
            plan_chunked_video(
                VideoRequest(width=1279, height=720, fps=16, total_seconds=5, chunk_seconds=5)
            )

    def test_increment_seed_policy_assigns_per_segment_seeds(self):
        plan = plan_chunked_video(
            VideoRequest(
                width=1280,
                height=720,
                fps=16,
                total_seconds=15,
                chunk_seconds=5,
                seed=500,
                seed_policy=SeedPolicy.INCREMENT,
            )
        )

        self.assertEqual([segment.seed for segment in plan.segments], [500, 501, 502])

    def test_segment_prompts_override_shared_prompt(self):
        plan = plan_chunked_video(
            VideoRequest(
                width=1280,
                height=720,
                fps=16,
                total_seconds=15,
                chunk_seconds=5,
                prompt="shared city drive",
                segment_prompts=("enter downtown", "", "cross the bridge"),
            )
        )

        self.assertEqual(
            [segment.prompt for segment in plan.segments],
            ["enter downtown", "shared city drive", "cross the bridge"],
        )


if __name__ == "__main__":
    unittest.main()
