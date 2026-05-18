from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_src_on_path() -> None:
    candidates = []
    env_repo = os.getenv("AGI_ARC_REPO")
    if env_repo:
        candidates.append(Path(env_repo) / "src")
    candidates.extend(
        [
            Path("/kaggle/working/agi-arc/src"),
            Path("/kaggle/input/agi-arc/src"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
            return


_ensure_repo_src_on_path()

from agents.agent import Agent
from arc_agi3.core.config import AgentConfig, LLMConfig
from arc_agi3.integrations.official_agent import OfficialAgentPolicy
from arcengine import FrameData, GameAction, GameState


class MyAgent(Agent):
    """Official ARC-AGI-3 agent wrapper used by the offline evaluation notebook."""

    MAX_ACTIONS = int(os.getenv("ARC_AGI3_MAX_ACTIONS", "500"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        agent_config = AgentConfig()
        agent_config.budget.max_steps_per_level = int(os.getenv("ARC_AGI3_MAX_STEPS", "128"))
        agent_config.budget.explore_phase_steps = int(os.getenv("ARC_AGI3_EXPLORE_STEPS", "24"))
        agent_config.budget.novelty_patience_steps = int(os.getenv("ARC_AGI3_NOVELTY_PATIENCE_STEPS", "6"))
        agent_config.budget.revisit_limit = int(os.getenv("ARC_AGI3_REVISIT_LIMIT", "3"))
        llm_config = LLMConfig(
            enabled=os.getenv("ARC_AGI3_LLM_ENABLED", "0") == "1",
            provider=os.getenv("ARC_AGI3_LLM_PROVIDER", "noop"),
            control_mode=os.getenv("ARC_AGI3_LLM_CONTROL_MODE", "directed"),
            model=os.getenv("ARC_AGI3_LLM_MODEL", ""),
            model_path=os.getenv("ARC_AGI3_LLM_MODEL_PATH", ""),
            device=os.getenv("ARC_AGI3_LLM_DEVICE", "auto"),
            dtype=os.getenv("ARC_AGI3_LLM_DTYPE", "auto"),
            start_step=int(os.getenv("ARC_AGI3_LLM_START_STEP", "8")),
            step_interval=int(os.getenv("ARC_AGI3_LLM_STEP_INTERVAL", "8")),
            max_calls_per_episode=int(os.getenv("ARC_AGI3_LLM_MAX_CALLS", "12")),
            max_new_tokens=int(os.getenv("ARC_AGI3_LLM_MAX_NEW_TOKENS", "256")),
            thinking_mode=os.getenv("ARC_AGI3_LLM_THINKING_MODE", "brief"),
            thinking_max_new_tokens=int(os.getenv("ARC_AGI3_LLM_THINKING_MAX_NEW_TOKENS", "512")),
            trace_enabled=os.getenv("ARC_AGI3_LLM_SHOW_TRACE", "0") == "1",
            trace_print_prompt=os.getenv("ARC_AGI3_LLM_SHOW_PROMPT", "0") == "1",
        )
        self.policy = OfficialAgentPolicy(agent_config=agent_config, llm_config=llm_config)
        self._printed_llm_traces = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        action = self.policy.choose_action(latest_frame)
        self._print_new_llm_traces()
        return action

    def _print_new_llm_traces(self) -> None:
        if os.getenv("ARC_AGI3_LLM_SHOW_TRACE", "0") != "1":
            return
        traces = self.policy.runtime.llm.recent_traces()
        if len(traces) < self._printed_llm_traces:
            self._printed_llm_traces = 0
        for trace in traces[self._printed_llm_traces :]:
            print(f"llm_trace_step={trace.step_idx} state={trace.state_key}")
            if os.getenv("ARC_AGI3_LLM_SHOW_PROMPT", "0") == "1":
                print("llm_prompt<<<")
                print(trace.prompt)
                print(">>>")
            if trace.raw_response:
                print("llm_raw_response<<<")
                print(trace.raw_response.strip())
                print(">>>")
            if trace.ranked_actions:
                rendered = [
                    f"{item.action.key} score={item.score:.2f} reason={item.reason}"
                    for item in trace.ranked_actions
                ]
                print("llm_ranked_actions=", rendered)
            if trace.next_test is not None:
                print(
                    "llm_next_test=",
                    f"{trace.next_test.key} conf={trace.next_test.confidence:.2f} reason={trace.next_test.rationale}",
                )
            if trace.directive is not None:
                preferred = trace.directive.preferred_action.key if trace.directive.preferred_action is not None else "none"
                print(
                    "llm_directive=",
                    f"goal={trace.directive.goal_key or 'none'} preferred={preferred} "
                    f"avoid={trace.directive.avoid_action_keys} commitment={trace.directive.commitment_steps} "
                    f"conf={trace.directive.confidence:.2f} summary={trace.directive.goal_summary}",
                )
        self._printed_llm_traces = len(traces)
