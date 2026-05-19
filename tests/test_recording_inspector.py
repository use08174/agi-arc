from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.inspect_recording_jsonl import load_recording_jsonl, normalize_frame_to_single_grid


class RecordingInspectorTest(unittest.TestCase):
    def test_normalize_frame_to_single_grid_uses_latest_nested_frame(self) -> None:
        grid = normalize_frame_to_single_grid([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]]])

        self.assertEqual(grid, [[0, 1], [0, 0]])

    def test_load_recording_jsonl_extracts_action_and_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.recording.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "timestamp": "t0",
                        "data": {
                            "game_id": "bp35",
                            "state": "IN_PROGRESS",
                            "levels_completed": 1,
                            "win_levels": 3,
                            "action": {"name": "ACTION6", "data": {"x": 1, "y": 4}},
                            "frame": [[[1, 0], [0, 2]]],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            frames = load_recording_jsonl(path)

        self.assertEqual(frames[0]["action"], "ACTION6|x=1,y=4")
        self.assertEqual(frames[0]["grid"], [[1, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()
