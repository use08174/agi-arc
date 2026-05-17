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
        frame_summary = self._summarize_frame(context)
        lines = [
            "You are helping an ARC-AGI-3 agent.",
            "Do not choose arbitrary actions. Prefer actions that test a concrete rule.",
            "Do not rush to an apparent goal if the environment may require a setup action first.",
            "Prefer prerequisite-changing actions when a direct goal action has already failed or led to dead ends.",
            "Avoid actions already known to end the episode in failure.",
            f"State key: {context.observation.state_key}",
            f"Status: {status}",
            f"Frame summary: {frame_summary}",
            f"Known promising actions: {', '.join(context.known_promising_actions) or 'none'}",
            f"Known dangerous actions: {', '.join(context.known_dangerous_actions) or 'none'}",
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

    def _summarize_frame(self, context: LLMContext) -> str:
        frame = context.observation.frame
        if frame is None or not frame.grid:
            return "no grid"

        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if grid[0] else 0
        nonzero: list[tuple[int, int, int]] = []
        colors: set[int] = set()
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                if value != 0:
                    value_int = int(value)
                    colors.add(value_int)
                    if len(nonzero) < 24:
                        nonzero.append((x, y, value_int))

        if nonzero:
            sample = ", ".join(f"({x},{y})={value}" for x, y, value in nonzero[:12])
        else:
            sample = "none"

        return (
            f"size={width}x{height}; "
            f"nonzero_count={sum(1 for row in grid for value in row if value != 0)}; "
            f"colors={sorted(colors) if colors else []}; "
            f"sample_nonzero={sample}"
        )
