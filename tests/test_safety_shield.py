from __future__ import annotations

import unittest

from arc_agi3.core.types import Action, Frame, GameStatus, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.planning.safety_shield import SafetyShield


class SafetyShieldTest(unittest.TestCase):
    def test_shield_prefers_non_meta_replacement_over_undo(self) -> None:
        memory = GameMemory()
        memory.remember_undo_like("ACTION7", "ACTION7")
        observation = Observation(
            state_key="s0",
            frame=Frame(grid=((0,),), status=GameStatus.IN_PROGRESS),
            changed=True,
        )

        replacement = SafetyShield().validate_or_replace(
            Action(name="ACTION7"),
            observation,
            [Action(name="ACTION7"), Action(name="ACTION3")],
            memory,
        )

        self.assertEqual(replacement.name, "ACTION3")


if __name__ == "__main__":
    unittest.main()
