import unittest

from wan_ltx_studio.server.dev_api import _plan_to_payload, _request_from_payload
from wan_ltx_studio.planning import plan_chunked_video


class DevApiPayloadTests(unittest.TestCase):
    def test_payload_uses_frontend_field_names(self):
        request = _request_from_payload(
            {
                "width": 1280,
                "height": 720,
                "totalSeconds": 15,
                "fps": 16,
                "chunkSeconds": 5,
                "seed": 42,
                "seedPolicy": "increment",
            }
        )
        response = _plan_to_payload(plan_chunked_video(request))

        self.assertEqual(response["targetTimelineFrames"], 240)
        self.assertEqual(response["actualOutputFrames"], 241)
        self.assertEqual(response["segments"][2]["seed"], 44)
        self.assertEqual(response["segments"][1]["continuity"]["source"], "previous_segment")


if __name__ == "__main__":
    unittest.main()
