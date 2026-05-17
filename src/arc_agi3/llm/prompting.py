from __future__ import annotations

from arc_agi3.llm.types import LLMContext


class PromptBuilder:
    """Converts agent state into a compact provider-neutral prompt.

    This is intentionally plain-text so we can feed the same content to
    an API model, local model, or offline transcript logger later.
    """

    def build(self, context: LLMContext) -> str:
        status = (
            context.observation.frame.status.value
            if context.observation.frame is not None
            else "UNKNOWN"
        )
        lines = [
            "You are helping an ARC-AGI-3 agent.",
            "Do not choose arbitrary actions. Prefer actions that test a concrete rule.",
            "Do not rush to an apparent goal if the environment may require a setup action first.",
            "Prefer prerequisite-changing actions when a direct goal action has already failed or led to dead ends.",
            f"State key: {context.observation.state_key}",
            f"Status: {status}",
            f"Known promising actions: {', '.join(context.known_promising_actions) or 'none'}",
            f"Recent states: {', '.join(context.recent_states[-8:]) or 'none'}",
            "Candidate actions:",
        ]
        for action in context.candidate_actions:
            lines.append(f"- {action.key}")
        if context.latest_transitions:
            lines.append("Recent transitions:")
            lines.extend(f"- {item}" for item in context.latest_transitions[-6:])
        if context.prior_hypotheses:
            lines.append("Prior hypotheses:")
            lines.extend(
                f"- {hyp.summary} (confidence={hyp.confidence:.2f})"
                for hyp in context.prior_hypotheses[-5:]
            )
        lines.append(
            "Return a ranked short list of actions and optional rule hypotheses."
        )
        return "\n".join(lines)
