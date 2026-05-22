from arc_agi3.core.types import Action, Frame, GameStatus, Observation
from arc_agi3.reasoning.scene_delta import GoalProgressScorer, SceneDeltaInterpreter


def test_scene_delta_interpreter_detects_object_motion():
    before = Observation(
        state_key="before",
        frame=Frame(
            grid=((0, 0, 0), (0, 2, 0), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"scene_goal_kinds": ["move_to_marker"]},
    )
    after = Observation(
        state_key="after",
        frame=Frame(
            grid=((0, 0, 0), (0, 0, 2), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={},
    )

    delta = SceneDeltaInterpreter().interpret(before, after, Action("ACTION4"))
    notes = delta.to_notes()

    assert notes["scene_delta_kind"] == "object_motion"
    assert notes["scene_delta_moved_count"] >= 1


def test_goal_progress_scorer_rewards_goal_aligned_delta():
    before = Observation(
        state_key="before",
        frame=Frame(
            grid=((0, 0, 0), (0, 2, 0), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"scene_goal_kinds": ["move_to_marker"]},
    )
    after = Observation(
        state_key="after",
        frame=Frame(
            grid=((0, 0, 0), (0, 0, 2), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={},
    )
    delta = SceneDeltaInterpreter().interpret(before, after, Action("ACTION4"))

    notes = GoalProgressScorer().score(
        before_notes=before.notes,
        after_notes=after.notes,
        delta=delta,
    )

    assert notes["scene_goal_progress_score"] > 0
    assert "movement_goal_object_motion" in notes["scene_goal_progress_reasons"]


def test_goal_progress_scorer_ignores_top_band_clicks():
    before = Observation(
        state_key="before",
        frame=Frame(
            grid=((0, 0, 0), (0, 2, 0), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"scene_goal_kinds": ["move_to_marker"]},
    )
    after = Observation(
        state_key="after",
        frame=Frame(
            grid=((0, 0, 0), (0, 0, 2), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"coordinate_click_role": "top_band"},
    )
    delta = SceneDeltaInterpreter().interpret(before, after, Action("ACTION6", {"x": 1, "y": 0}))

    notes = GoalProgressScorer().score(
        before_notes=before.notes,
        after_notes=after.notes,
        delta=delta,
    )

    assert notes["scene_goal_progress_score"] == 0.0
    assert notes["scene_goal_progress_reasons"] == ["top_band_click_not_goal_progress"]
