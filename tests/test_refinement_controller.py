from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Frame, GameStatus, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.policy.refinement_controller import RefinementController


def test_refinement_controller_prefers_movement_for_navigation_prior():
    observation = Observation(
        state_key="nav",
        frame=Frame(
            grid=((0, 1, 0), (0, 0, 0), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={
            "external_task_family": "navigation",
            "compressarc_in_out_same_size": True,
            "compressarc_all_in_same_size": True,
        },
    )
    actions = [Action("ACTION5"), Action("ACTION2"), Action("ACTION6", {"x": 1, "y": 1})]

    decision = RefinementController(AgentConfig()).choose_action(
        observation=observation,
        actions=actions,
        graph=StateGraph(),
        game_memory=GameMemory(),
        recent_action_keys=[],
    )

    assert decision is not None
    assert decision.action.name == "ACTION2"
    assert decision.source == "refinement"


def test_refinement_controller_can_lower_editing_prior_to_click():
    observation = Observation(
        state_key="edit",
        frame=Frame(
            grid=(
                (0, 1, 1, 0),
                (0, 1, 0, 2),
                (3, 0, 4, 2),
            ),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={"external_task_family": "editing"},
    )
    actions = [Action("ACTION1"), Action("ACTION6", {"x": 1, "y": 0}), Action("ACTION6", {"x": 3, "y": 2})]

    decision = RefinementController(AgentConfig()).choose_action(
        observation=observation,
        actions=actions,
        graph=StateGraph(),
        game_memory=GameMemory(),
        recent_action_keys=["ACTION1"],
    )

    assert decision is not None
    assert decision.action.name == "ACTION6"
    assert decision.action.payload


def test_refinement_controller_uses_coordinate_only_strategy():
    observation = Observation(
        state_key="coord",
        frame=Frame(
            grid=(
                (0, 0, 0, 0),
                (0, 2, 2, 0),
                (0, 2, 0, 0),
                (0, 0, 0, 0),
            ),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={},
    )
    actions = [Action("ACTION6", {"x": 1, "y": 1}), Action("ACTION6", {"x": 3, "y": 3})]

    decision = RefinementController(AgentConfig()).choose_action(
        observation=observation,
        actions=actions,
        graph=StateGraph(),
        game_memory=GameMemory(),
        recent_action_keys=[],
    )

    assert decision is not None
    assert decision.action.name == "ACTION6"
    assert "coordinate_only" in decision.reason
    assert any(goal in decision.reason for goal in ("paint_reference", "match_shape", "explore_causal_effects"))


def test_refinement_controller_uses_movement_only_strategy():
    observation = Observation(
        state_key="move",
        frame=Frame(
            grid=((0, 1, 0), (0, 0, 0), (0, 0, 0)),
            status=GameStatus.IN_PROGRESS,
        ),
        changed=True,
        notes={},
    )
    actions = [Action("ACTION1"), Action("ACTION2"), Action("ACTION3"), Action("ACTION4")]

    decision = RefinementController(AgentConfig()).choose_action(
        observation=observation,
        actions=actions,
        graph=StateGraph(),
        game_memory=GameMemory(),
        recent_action_keys=[],
    )

    assert decision is not None
    assert decision.action.name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    assert "movement_only" in decision.reason


def test_refinement_controller_prefers_learned_move_toward_nav_path():
    observation = Observation(
        state_key="nav-path",
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
    memory = GameMemory()
    memory.learned_action_semantics.meaning_for("ACTION4").move_vectors[(1, 0)] = 3
    memory.learned_action_semantics.meaning_for("ACTION2").move_vectors[(0, 1)] = 3
    actions = [Action("ACTION1"), Action("ACTION2"), Action("ACTION3"), Action("ACTION4")]

    decision = RefinementController(AgentConfig()).choose_action(
        observation=observation,
        actions=actions,
        graph=StateGraph(),
        game_memory=memory,
        recent_action_keys=["ACTION1", "ACTION3"],
    )

    assert decision is not None
    assert decision.action.name in {"ACTION2", "ACTION4"}
    assert observation.notes["nav_has_path"] is True
