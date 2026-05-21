from __future__ import annotations

from dataclasses import dataclass

from arc_agi3.abstraction.state_lifter import StateLifter
from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.policy.action_signature import classify_runtime_signature
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
        signature = classify_runtime_signature(actions, observation)
        lifted.notes.update(signature.to_notes())
        lifted.notes.update(self._probe_phase_notes(signature.action_names, recent_action_keys))
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
            reason=f"{signature.control_family}/{top}: {reason}",
        )

    def _probe_phase_notes(self, action_names: tuple[str, ...], recent_action_keys: list[str]) -> dict[str, object]:
        tried_names = {key.split("|", 1)[0] for key in recent_action_keys}
        movement_names = {name for name in action_names if name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}}
        mode_names = {name for name in action_names if name in {"ACTION5", "ACTION7"}}
        movement_probe_done = bool(movement_names) and movement_names.issubset(tried_names)
        mode_probe_done = not mode_names or mode_names.issubset(tried_names)
        movement_probe_count = sum(1 for key in recent_action_keys if key.split("|", 1)[0] in movement_names)
        coordinate_probe_count = sum(1 for key in recent_action_keys if key.startswith("ACTION6|"))
        return {
            "runtime_movement_probe_done": movement_probe_done,
            "runtime_mode_probe_done": mode_probe_done,
            "runtime_movement_probe_count": movement_probe_count,
            "runtime_coordinate_probe_count": coordinate_probe_count,
        }
