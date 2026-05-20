from __future__ import annotations

from collections import deque

from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import DecisionTrace, Observation
from arc_agi3.exploration.frontier import FrontierExplorer
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.perception.hasher import StateHasher
from arc_agi3.planning.experiment_policy import ExperimentPolicy
from arc_agi3.planning.experiment_runner import ExperimentRunner
from arc_agi3.planning.simple_planner import SimplePlanner


class ArcAgentRuntime:
    def __init__(
        self,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.hasher = StateHasher()
        self.explorer = FrontierExplorer(self.config)
        self.planner = SimplePlanner()
        self.experiment_policy = ExperimentPolicy()
        self.experiment_runner = ExperimentRunner()
        self.game_memory = GameMemory()
        self.graph = StateGraph()
        self.previous_observation: Observation | None = None
        self.recent_states: deque[str] = deque(
            maxlen=self.config.budget.recent_state_window
        )
        self.recent_actions: deque[str] = deque(
            maxlen=self.config.budget.recent_action_window
        )
        self.recent_action_keys: deque[str] = deque(
            maxlen=self.config.budget.recent_action_window
        )
        self.recent_action_families: deque[str] = deque(
            maxlen=self.config.budget.recent_action_window
        )
        self.recent_effect_transforms: deque[str] = deque(
            maxlen=self.config.budget.recent_action_window
        )
        self.recent_progress_scores: deque[float] = deque(
            maxlen=self.config.budget.recent_action_window
        )
        self.decision_traces: list[DecisionTrace] = []
        self.steps_since_new_state = 0
        self.steps_since_semantic_progress = 0
        self.click_no_progress_counts: dict[str, int] = {}
        self.previous_action_context: dict[str, object] | None = None
        self.movement_commitment_action_key: str | None = None
        self.movement_commitment_remaining = 0

    def reset_level(self) -> None:
        self.graph = StateGraph()
        self.previous_observation = None
        self.recent_states = deque(maxlen=self.config.budget.recent_state_window)
        self.recent_actions = deque(maxlen=self.config.budget.recent_action_window)
        self.recent_action_keys = deque(maxlen=self.config.budget.recent_action_window)
        self.recent_action_families = deque(maxlen=self.config.budget.recent_action_window)
        self.recent_effect_transforms = deque(maxlen=self.config.budget.recent_action_window)
        self.recent_progress_scores = deque(maxlen=self.config.budget.recent_action_window)
        self.decision_traces = []
        self.steps_since_new_state = 0
        self.steps_since_semantic_progress = 0
        self.click_no_progress_counts = {}
        self.previous_action_context = None
        self.movement_commitment_action_key = None
        self.movement_commitment_remaining = 0
