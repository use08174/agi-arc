from __future__ import annotations

from dataclasses import dataclass, field

from arc_agi3.core.types import Action, Observation, RankedAction, RuleHypothesis


@dataclass(slots=True)
class LLMContext:
    observation: Observation
    candidate_actions: list[Action]
    recent_states: list[str]
    known_promising_actions: list[str]
    latest_transitions: list[str] = field(default_factory=list)
    prior_hypotheses: list[RuleHypothesis] = field(default_factory=list)


@dataclass(slots=True)
class LLMDecisionBundle:
    ranked_actions: list[RankedAction] = field(default_factory=list)
    hypotheses: list[RuleHypothesis] = field(default_factory=list)
    raw_response: str = ""
