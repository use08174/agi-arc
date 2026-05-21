from __future__ import annotations

from dataclasses import dataclass

from arc_agi3.abstraction.state_lifter import StateLifter
from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.policy.action_realizer import ActionRealizer
from arc_agi3.reasoning.critic import HypothesisCritic
from arc_agi3.reasoning.refiner import HypothesisRefiner
from arc_agi3.reasoning.verifier import HypothesisVerifier


@dataclass(slots=True)
class RefinementDecision:
    action: Action
    source: str
    reason: str


class RefinementController:
    """Poetiq-style generate/critique/refine/verify controller."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.lifter = StateLifter()
        self.refiner = HypothesisRefiner()
        self.critic = HypothesisCritic()
        self.verifier = HypothesisVerifier()
        self.realizer = ActionRealizer()

    def choose_action(
        self,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> RefinementDecision | None:
        if not actions:
            return None
        lifted = self.lifter.lift(observation)
        hypotheses = self.refiner.generate(lifted)
        hypotheses = self.critic.critique(hypotheses, lifted)
        hypotheses = self.refiner.refine(hypotheses)
        intents = self.verifier.intents_for(hypotheses, lifted)
        realized = self.realizer.realize(
            intents=intents,
            actions=actions,
            observation=observation,
            graph=graph,
            game_memory=game_memory,
            recent_action_keys=recent_action_keys,
        )
        if realized is None:
            return None
        action, reason = realized
        top = hypotheses[0].kind.value if hypotheses else "unknown"
        return RefinementDecision(
            action=action,
            source="refinement",
            reason=f"{top}: {reason}",
        )
