from __future__ import annotations

import unittest

from arc_agi3.runner.offline_eval import extract_scorecard_json


class OfflineEvalTest(unittest.TestCase):
    def test_extract_scorecard_json_from_final_report_lines(self) -> None:
        scorecard = extract_scorecard_json(
            [
                "2026-05-18 00:00:00 | INFO | --- FINAL SCORECARD REPORT ---",
                "2026-05-18 00:00:00 | INFO | {",
                '2026-05-18 00:00:00 | INFO |   "environments": [',
                '2026-05-18 00:00:00 | INFO |     {"id": "bp35-1", "score": 0.5}',
                "2026-05-18 00:00:00 | INFO |   ]",
                "2026-05-18 00:00:00 | INFO | }",
            ]
        )

        self.assertIsNotNone(scorecard)
        self.assertEqual(scorecard["environments"][0]["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
