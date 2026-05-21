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
        recent_action_keys=[],
    )

    assert decision is not None
    assert decision.action.name == "ACTION6"
    assert decision.action.payload
