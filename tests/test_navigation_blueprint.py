from arc_agi3.abstraction.navigation_blueprint import NavigationBlueprintBuilder
from arc_agi3.abstraction.state_lifter import StateLifter
from arc_agi3.core.types import Frame, GameStatus, Observation


def test_navigation_blueprint_finds_free_space_path():
    observation = Observation(
        state_key="nav",
        frame=Frame(
            grid=(
                (1, 1, 1, 1, 1),
                (1, 2, 0, 0, 1),
                (1, 1, 1, 0, 1),
                (1, 3, 0, 0, 1),
                (1, 1, 1, 1, 1),
            ),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"semantic_player_pos": (1, 1)},
    )

    lifted = StateLifter().lift(observation)
    nav = NavigationBlueprintBuilder().build(lifted)
    notes = nav.to_notes()

    assert notes["nav_has_path"] is True
    assert notes["nav_path_length"] > 0
    assert notes["nav_next_step_delta"] in {(1, 0), (0, 1)}
    assert notes["nav_nearest_target"] is not None
