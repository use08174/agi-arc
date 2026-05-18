from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import Action, ExperimentProposal, Observation, RankedAction, RuleHypothesis


@dataclass(slots=True)
class LLMContext:
    observation: Observation
    candidate_actions: list[Action]
    recent_states: list[str]
    known_promising_actions: list[str]
    known_dangerous_actions: list[str] = field(default_factory=list)
    known_restart_like_actions: list[str] = field(default_factory=list)
    known_undo_like_actions: list[str] = field(default_factory=list)
    known_failure_revert_actions: list[str] = field(default_factory=list)
    candidate_action_evidence: list[str] = field(default_factory=list)
    learned_action_semantics: list[str] = field(default_factory=list)
    latest_transitions: list[str] = field(default_factory=list)
    prior_hypotheses: list[RuleHypothesis] = field(default_factory=list)
    semantic_ascii_map: str = ""
    world_model_summary: list[str] = field(default_factory=list)
    recent_scene_events: list[str] = field(default_factory=list)
    goal_hypotheses: list[str] = field(default_factory=list)
    relation_candidates: list[str] = field(default_factory=list)
    proposed_tests: list[str] = field(default_factory=list)
    candidate_subgoals: list[dict[str, Any]] = field(default_factory=list)
    available_experiments: list[ExperimentProposal] = field(default_factory=list)
    experiment_history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LLMDecisionBundle:
    ranked_actions: list[RankedAction] = field(default_factory=list)
    hypotheses: list[RuleHypothesis] = field(default_factory=list)
    next_test: ExperimentProposal | None = None
    raw_response: str = ""
