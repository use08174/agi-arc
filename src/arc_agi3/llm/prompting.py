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
        if context.candidate_action_evidence:
            lines.append("Candidate action evidence:")
            lines.extend(f"- {item}" for item in context.candidate_action_evidence[:12])
        repeated_motifs = context.observation.notes.get("repeated_motif_summary", [])
        if repeated_motifs:
            lines.append("Repeated motif candidates:")
            for motif in repeated_motifs[:4]:
                lines.append(
                    f"- count={motif['count']} area={motif['area']} region={motif['region']} bbox={motif['bbox']}"
                )
        anchors = context.observation.notes.get("anchor_region_summary", [])
        if anchors:
            lines.append("Anchor region candidates:")
            for anchor in anchors[:4]:
                lines.append(
                    f"- anchor={anchor['anchor']} area={anchor['area']} region={anchor['region']} color={anchor['color']} bbox={anchor['bbox']}"
                )
        anchor_patches = context.observation.notes.get("anchor_patch_summary", [])
        if anchor_patches:
            lines.append("Anchor patch states:")
            for patch in anchor_patches[:4]:
                lines.append(
                    f"- anchor={patch['anchor']} region={patch['region']} nonzero={patch['nonzero']} colors={patch['colors']} signature={patch['signature']}"
                )
        region_changes = context.observation.notes.get("region_change_summary", [])
        if region_changes:
            lines.append("Recent region-structure changes:")
            lines.extend(f"- {item}" for item in region_changes[:6])
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
            "Prefer unexplored actions over repeated zero-reward actions unless there is direct evidence of progress."
        )
        lines.append(
            "Do not use nonzero pixel count as a goal by itself unless reward or win evidence supports it."
        )
        lines.append(
            "Look for repeated motifs, anchor panels, and state indicators that may need to match a target before finishing."
        )
        lines.append(
            "If corner or HUD patches have distinct signatures, consider whether one patch is a target and another is a mutable state that should be aligned."
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
            f"playfield_nonzero={context.observation.notes.get('playfield_nonzero_count', 'n/a')}; "
            f"hud_nonzero={context.observation.notes.get('hud_nonzero_count', 'n/a')}; "
            f"hud_rows_hint={context.observation.notes.get('hud_rows_hint', 'n/a')}; "
            f"colors={sorted(colors) if colors else []}; "
            f"sample_nonzero={sample}"
        )
