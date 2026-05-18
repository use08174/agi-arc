from __future__ import annotations

from collections import deque

from arc_agi3.core.config import AgentConfig, LLMConfig
from arc_agi3.core.types import DecisionTrace, Observation
from arc_agi3.exploration.frontier import FrontierExplorer
from arc_agi3.llm.factory import build_llm_provider
from arc_agi3.llm.manager import LLMHookManager
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.perception.hasher import StateHasher
from arc_agi3.planning.simple_planner import SimplePlanner


class ArcAgentRuntime:
    def __init__(
        self,
        config: AgentConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm_config = llm_config or LLMConfig()
        self.hasher = StateHasher()
        self.explorer = FrontierExplorer(self.config)
        self.planner = SimplePlanner()
        self.llm = LLMHookManager(self.llm_config, build_llm_provider(self.llm_config))
        self.game_memory = GameMemory()
        self.graph = StateGraph()
        self.previous_observation: Observation | None = None
        self.recent_states: deque[str] = deque(
            maxlen=self.config.budget.recent_state_window
        )
        self.recent_actions: deque[str] = deque(
            maxlen=self.config.budget.recent_action_window
        )
        self.decision_traces: list[DecisionTrace] = []
        self.steps_since_new_state = 0
        self.steps_since_semantic_progress = 0

    def reset_level(self) -> None:
        self.graph = StateGraph()
        self.previous_observation = None
        self.recent_states = deque(maxlen=self.config.budget.recent_state_window)
        self.recent_actions = deque(maxlen=self.config.budget.recent_action_window)
        self.decision_traces = []
        self.steps_since_new_state = 0
        self.steps_since_semantic_progress = 0
        self.llm.reset_episode()

    def close(self) -> None:
        self.llm.close()
