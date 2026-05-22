from arc_agi3.abstraction.scene_blueprint import SceneBlueprintBuilder
from arc_agi3.abstraction.state_lifter import StateLifter
from arc_agi3.core.types import Frame, GameStatus, Observation
from arc_agi3.reasoning.goal_inference import GoalInferencer, GoalKind


def test_scene_blueprint_finds_repeated_shapes_and_goals():
    observation = Observation(
        state_key="scene",
        frame=Frame(
            grid=(
                (0, 0, 0, 0, 0, 0),
                (0, 2, 2, 0, 3, 3),
                (0, 2, 0, 0, 3, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 4, 0, 0, 4, 0),
                (0, 0, 0, 0, 0, 0),
            ),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"runtime_control_family": "mixed_move_coordinate"},
    )

    lifted = StateLifter().lift(observation)
    blueprint = SceneBlueprintBuilder().build(lifted)
    goals = GoalInferencer().infer(blueprint)

    assert blueprint.repeated_shape_groups
    assert "same_shape_group" in blueprint.to_notes()["scene_relations"]
    assert blueprint.goal_targets
    assert any(goal.kind in {GoalKind.MATCH_SHAPE, GoalKind.PAINT_REFERENCE} for goal in goals)
