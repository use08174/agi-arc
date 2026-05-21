from arc_agi3.abstraction.state_lifter import StateLifter
from arc_agi3.core.types import Frame, GameStatus, Observation


def test_state_lifter_extracts_objects_and_click_candidates():
    observation = Observation(
        state_key="demo",
        frame=Frame(
            grid=(
                (0, 1, 1, 0),
                (0, 1, 0, 2),
                (0, 0, 0, 2),
            ),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"external_task_family": "editing"},
    )

    lifted = StateLifter().lift(observation)

    assert lifted.width == 4
    assert lifted.height == 3
    assert len(lifted.objects) == 2
    assert lifted.objects[0].color == 1
    assert lifted.objects[0].area == 3
    assert (1, 0) in lifted.candidate_clicks
    assert lifted.notes["external_task_family"] == "editing"
