from __future__ import annotations

import unittest

from arc_agi3.integrations.official_agent import normalize_action_name


class OfficialAgentTest(unittest.TestCase):
    def test_numeric_available_actions_are_normalized(self) -> None:
        self.assertEqual(normalize_action_name(3), "ACTION3")
        self.assertEqual(normalize_action_name("6"), "ACTION6")

    def test_named_available_actions_are_preserved(self) -> None:
        self.assertEqual(normalize_action_name("ACTION4"), "ACTION4")
        self.assertEqual(normalize_action_name("reset"), "RESET")


if __name__ == "__main__":
    unittest.main()
