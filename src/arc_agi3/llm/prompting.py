from __future__ import annotations

from arc_agi3.llm.types import LLMContext


class PromptBuilder:
    """Converts agent state into a world-model prompt."""

    def build(self, context: LLMContext) -> str:
        status = context.observation.frame.status.value if context.observation.frame is not None else "UNKNOWN"
        frame_summary = self._summarize_frame(context)
        lines = [
            "You are helping an ARC-AGI-3 agent. The symbolic planner owns final movement; you provide ranking and rule hypotheses.",
            "Use the semantic map and world model. Do not choose arbitrary unexplored actions.",
            "Never prefer actions marked unsafe, deadly, blocked, HUD-only, feedback-only, noop-loop, terminal-loss, or RESTART_LIKE.",
            "Treat UNDO_LIKE as a meta action, not a normal candidate for forward progress.",
            "If recent behavior is looping without reward or new states, prefer a concrete unseen test over repeating a merely safe route.",
            "When several safe moves repeat without semantic progress, propose a counterfactual test that could falsify the current plan.",
            "If an active experiment is already listed, continue it unless the evidence has contradicted or completed it; do not switch goals every step.",
            "Never prefer actions marked DISPLAY_LIKE_CLICK unless there is direct evidence that the display itself is interactive.",
            "Prefer actions that move along a safe path to items/goals/buttons, or click objects matching learned object rules.",
            "State key: " + context.observation.state_key,
            "Status: " + status,
            "Frame summary: " + frame_summary,
        ]
        if context.semantic_ascii_map:
            lines.append("Semantic ASCII map legend: P=player #=wall .=floor/unknown i=item G=goal B=button H=hazard")
            lines.append(context.semantic_ascii_map)
        if context.world_model_summary:
            lines.append("World model summary:")
            lines.extend("- " + item for item in context.world_model_summary[:12])
        if context.learned_action_semantics:
            lines.append("Learned action semantics from direct experiments:")
            lines.extend("- " + item for item in context.learned_action_semantics[:8])
        if context.recent_scene_events:
            lines.append("Recent scene events:")
            lines.extend("- " + item for item in context.recent_scene_events[-10:])
        if context.goal_hypotheses:
            lines.append("Goal hypothesis families:")
            lines.extend("- " + item for item in context.goal_hypotheses[:6])
        if context.relation_candidates:
            lines.append("Relation candidates:")
            lines.extend("- " + item for item in context.relation_candidates[:6])
        if context.proposed_tests:
            lines.append("High-value tests proposed by hypothesis families:")
            lines.extend("- " + item for item in context.proposed_tests[:6])
        if context.experiment_history:
            lines.append("Experiment history:")
            lines.extend("- " + item for item in context.experiment_history[-6:])
        if context.available_experiments:
            lines.append("Available executable experiments:")
            for experiment in context.available_experiments[:12]:
                lines.append(
                    f"- {experiment.key}: kind={experiment.kind}; target={experiment.target}; "
                    f"rationale={experiment.rationale}; expected={experiment.expected_if_true}; failure={experiment.failure_signal}"
                )
        if context.candidate_subgoals:
            lines.append("Candidate subgoals:")
            lines.extend(f"- {item}" for item in context.candidate_subgoals[:8])
        lines.extend(
            [
                f"Known promising action keys: {', '.join(context.known_promising_actions) or 'none'}",
                f"Known dangerous action keys: {', '.join(context.known_dangerous_actions) or 'none'}",
                f"Known restart-like action keys: {', '.join(context.known_restart_like_actions) or 'none'}",
                f"Known undo-like action keys: {', '.join(context.known_undo_like_actions) or 'none'}",
                f"Known failure-revert action keys: {', '.join(context.known_failure_revert_actions) or 'none'}",
                f"Recent states: {', '.join(context.recent_states[-8:]) or 'none'}",
                "Candidate actions:",
            ]
        )
        for action in context.candidate_actions:
            lines.append(f"- {action.key}")
        if context.candidate_action_evidence:
            lines.append("Candidate action evidence:")
            lines.extend(f"- {item}" for item in context.candidate_action_evidence[:16])
        if context.latest_transitions:
            lines.append("Recent transitions:")
            lines.extend(f"- {item}" for item in context.latest_transitions[-6:])
        if context.prior_hypotheses:
            lines.append("Prior hypotheses:")
            lines.extend(f"- {hyp.summary} (confidence={hyp.confidence:.2f})" for hyp in context.prior_hypotheses[-5:])
        lines.extend(
            [
                "Important: HUD/counter/action-limit-bar changes alone are not progress.",
                "Do not assume a static-looking panel is a goal. Infer roles from events: disappearance, movement, patch changes, feedback, and relations.",
                "A plausible task may be collect->state-change->goal-gate, connect-same-color-regions, align/match panels, or plain reach-goal.",
                "Use affordances: blocking candidates may need to be broken, pushed, toggled, or otherwise changed before a route opens.",
                "If an object type disappeared after a click, infer an object rule such as click_removes and apply it to similar objects.",
                "If feedback flashes occur near a goal attempt, treat direct goal entry as possibly premature until a precondition is tested.",
                "If same-color regions are disconnected, consider whether connecting them is the latent objective.",
                "Use exact action keys including coordinates, for example ACTION6|x=32,y=32.",
                "Choose next_test from Available executable experiments when one can reduce uncertainty. Prefer high-information tests over safe repetition.",
                "After any brief internal reasoning, finish with one short JSON object only; no markdown or prose outside the final JSON.",
            ]
        )
        return "\n".join(lines)

    def _summarize_frame(self, context: LLMContext) -> str:
        frame = context.observation.frame
        if frame is None or not frame.grid:
            return "no grid"
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if grid[0] else 0
        colors = sorted({int(value) for row in grid for value in row if value != 0})
        return (
            f"size={width}x{height}; nonzero_count={sum(1 for row in grid for value in row if value != 0)}; "
            f"playfield_nonzero={context.observation.notes.get('playfield_nonzero_count', 'n/a')}; "
            f"hud_nonzero={context.observation.notes.get('hud_nonzero_count', 'n/a')}; "
            f"colors={colors}"
        )
