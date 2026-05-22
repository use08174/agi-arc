from arc_agi3.abstraction.object_ops import AbstractIntent, ObjectOp
from arc_agi3.core.types import Action, Frame, GameStatus, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.policy.coordinate_policy import CoordinateInteractionPolicy


def test_coordinate_policy_limits_top_band_after_budget():
    observation = Observation(
        state_key="coord",
        frame=Frame(
            grid=tuple(tuple(0 for _ in range(10)) for _ in range(10)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"runtime_control_family": "coordinate_only", "scene_goal_targets": [(5, 7)]},
    )
    actions = [
        Action("ACTION6", {"x": 5, "y": 1}),
        Action("ACTION6", {"x": 5, "y": 7}),
    ]
    memory = GameMemory()
    memory.coordinate_top_band_uses = 3

    ranked = CoordinateInteractionPolicy().rank(
        actions=actions,
        intent=AbstractIntent(ObjectOp.PROBE_CLICK, target=((5, 1), (5, 7))),
        observation=observation,
        game_memory=memory,
        recent_action_keys=[],
    )

    assert ranked[0].payload == {"x": 5, "y": 7}


def test_coordinate_policy_cools_failed_coordinate():
    memory = GameMemory()
    memory.remember_coordinate_click("ACTION6|x=5,y=7", "workspace_empty", 7, success=False)
    observation = Observation(
        state_key="coord",
        frame=Frame(
            grid=tuple(tuple(0 for _ in range(10)) for _ in range(10)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"runtime_control_family": "coordinate_only", "scene_goal_targets": [(5, 7), (6, 7)]},
    )
    actions = [
        Action("ACTION6", {"x": 5, "y": 7}),
        Action("ACTION6", {"x": 6, "y": 7}),
    ]

    ranked = CoordinateInteractionPolicy().rank(
        actions=actions,
        intent=AbstractIntent(ObjectOp.PROBE_CLICK, target=((5, 7), (6, 7))),
        observation=observation,
        game_memory=memory,
        recent_action_keys=[],
    )

    assert ranked[0].payload == {"x": 6, "y": 7}
