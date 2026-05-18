from __future__ import annotations

from arc_agi3.agents.base import ArcAgentRuntime
from arc_agi3.agents.env_understanding_agent import EnvUnderstandingAgent
from arc_agi3.core.types import Action, DecisionTrace, Observation, Transition
from arc_agi3.envs.base import ArcEnvironment
from arc_agi3.planning.safety_shield import SafetyShield


class GraphSearchAgent(ArcAgentRuntime):
    """Explore-first agent scaffold with semantic world-model updates."""

    def run_episode(self, env: ArcEnvironment) -> tuple[bool, int]:
        self.last_episode_end_reason = "not_started"
        self.last_episode_final_status = "UNKNOWN"
        self.last_episode_final_info = {}
        self.reset_level()
        self.safety_shield = SafetyShield()
        self.understanding_agent = EnvUnderstandingAgent()
        frame = env.reset()
        observation = self.hasher.observe(frame)
        self.game_memory.world_model.update_from_observation(observation.notes)
        self.understanding_agent.inspect(env, observation, self.game_memory)
        self.graph.touch(observation.state_key, terminal=False)
        self.initial_state_key = observation.state_key
        self.recent_states.append(observation.state_key)
        for step_idx in range(self.config.budget.max_steps_per_level):
            actions = env.valid_actions()
            action = self._choose_action(step_idx, observation, actions)
            action = self.safety_shield.validate_or_replace(action, observation, actions, self.game_memory)
            before_notes = dict(observation.notes)
            result = env.step(action)
            next_observation = self.hasher.observe(result.frame, previous=observation.frame)
            discovered_new_state = next_observation.state_key not in self.graph.nodes
            self.game_memory.world_model.update_from_observation(next_observation.notes)
            transition = Transition(
                from_state=observation.state_key,
                action=action,
                to_state=next_observation.state_key,
                changed=next_observation.changed,
                reward_delta=result.reward_delta,
                terminal=result.done,
                won=result.won,
                notes=next_observation.notes,
            )
            self.graph.record(transition)
            self.game_memory.world_model.learn_transition(
                action=action,
                before_notes=before_notes,
                after_notes=next_observation.notes,
                terminal_loss=result.done and not result.won,
            )
            experiment_outcome = self.game_memory.experiments.observe_transition(
                transition=transition,
                after_notes=next_observation.notes,
                world=self.game_memory.world_model,
            )
            if experiment_outcome is not None:
                self.game_memory.world_model.hypothesis_library.apply_experiment_result(
                    kind=experiment_outcome.proposal.kind,
                    status=experiment_outcome.status,
                    evidence=experiment_outcome.evidence,
                )
            self._learn_from_transition(transition)
            self._learn_meta_action(transition)
            self._learn_action_semantics(transition)
            self.recent_actions.append(action.name)
            if discovered_new_state:
                self.steps_since_new_state = 0
            else:
                self.steps_since_new_state += 1
            semantic_progress = bool(
                result.reward_delta > 0
                or next_observation.notes.get("collectible_progress", False)
                or next_observation.notes.get("anchor_patch_changes", [])
                or (experiment_outcome is not None and experiment_outcome.status == "confirmed")
            )
            if semantic_progress:
                self.steps_since_semantic_progress = 0
            else:
                self.steps_since_semantic_progress += 1
            if result.done:
                self.last_episode_end_reason = "environment_done"
                self.last_episode_final_status = result.frame.status.value
                self.last_episode_final_info = dict(result.frame.info)
                return result.won, step_idx + 1
            observation = next_observation
            self.recent_states.append(observation.state_key)
        self.last_episode_end_reason = "step_budget_exhausted"
        self.last_episode_final_status = observation.frame.status.value
        self.last_episode_final_info = dict(observation.frame.info)
        return False, self.config.budget.max_steps_per_level

    def _choose_action(self, step_idx: int, observation: Observation, actions: list[Action]) -> Action:
        actions = self._filter_useless_actions(observation, actions)
        force_exploration = self._should_force_exploration(step_idx, observation)
        ranked_actions = self.llm.rank_actions(
            observation=observation,
            candidate_actions=actions,
            graph=self.graph,
            game_memory=self.game_memory,
            recent_states=list(self.recent_states),
            step_idx=step_idx,
            trigger_reason="stalled" if force_exploration and step_idx >= self.config.budget.explore_phase_steps else "",
        )
        actions = self.explorer.reorder_with_rankings(
            observation=observation,
            actions=actions,
            ranked_actions=ranked_actions,
            graph=self.graph,
            game_memory=self.game_memory,
            force_exploration=force_exploration,
        )
        if self._should_force_counterfactual_exploration():
            counterfactual = self.explorer.choose_counterfactual_action(
                observation=observation,
                actions=actions,
                graph=self.graph,
                game_memory=self.game_memory,
                recent_action_names=list(self.recent_actions),
            )
            if counterfactual is not None:
                return self._trace_decision(step_idx, observation, counterfactual, "counterfactual", "breaking repetitive low-progress action loop")
        experiment_plan = self.planner.build_experiment_plan(
            experiment=self.game_memory.experiments.active,
            actions=actions,
            game_memory=self.game_memory,
        )
        if experiment_plan:
            step = experiment_plan[0]
            return self._trace_decision(step_idx, observation, step.action, "experiment", step.reason)
        if not force_exploration:
            plan = self.planner.build_plan(
                observation=observation,
                actions=actions,
                graph=self.graph,
                game_memory=self.game_memory,
                recent_states=set(self.recent_states),
            )
            if plan:
                step = plan[0]
                return self._trace_decision(step_idx, observation, step.action, "planner", step.reason)
        action = self.explorer.choose_action(
            observation=observation,
            actions=actions,
            graph=self.graph,
            game_memory=self.game_memory,
            recent_states=set(self.recent_states),
            force_exploration=force_exploration,
        )
        if action is None:
            fallback = self._fallback_action(observation, actions)
            return self._trace_decision(step_idx, observation, fallback, "fallback", "no ranked frontier action")
        return self._trace_decision(step_idx, observation, action, "explorer", "frontier selection")

    def _trace_decision(self, step_idx: int, observation: Observation, action: Action, source: str, reason: str) -> Action:
        self.decision_traces.append(
            DecisionTrace(
                step_idx=step_idx,
                state_key=observation.state_key,
                action=action,
                source=source,
                reason=reason,
            )
        )
        del self.decision_traces[:-64]
        return action

    def _should_force_counterfactual_exploration(self) -> bool:
        if self.steps_since_semantic_progress < self.config.budget.semantic_patience_steps:
            return False
        if len(self.recent_actions) < 3:
            return False
        return len(set(self.recent_actions)) <= 2

    def _should_force_exploration(self, step_idx: int, observation: Observation) -> bool:
        if step_idx < self.config.budget.explore_phase_steps:
            return True
        if self.steps_since_new_state >= self.config.budget.novelty_patience_steps:
            return True
        recent_arrivals = sum(1 for state in self.recent_states if state == observation.state_key)
        if recent_arrivals >= self.config.budget.revisit_limit:
            return True
        return False

    def _learn_from_transition(self, transition: Transition) -> None:
        notes = transition.notes
        likely_feedback_flash = bool(notes.get("likely_feedback_flash", False))
        changed_playfield_cells = int(notes.get("changed_playfield_cells", 0) or 0)
        changed_hud_cells = int(notes.get("changed_hud_cells", 0) or 0)
        collectible_progress = bool(notes.get("collectible_progress", False))
        semantic_player_moved = bool(notes.get("semantic_player_moved", False))
        hud_only_change = changed_hud_cells > 0 and changed_playfield_cells == 0 and transition.reward_delta <= 0 and not collectible_progress
        meaningful_change = changed_playfield_cells > 0 or transition.reward_delta > 0 or collectible_progress or semantic_player_moved
        if likely_feedback_flash or hud_only_change:
            self.game_memory.remember_effect(transition.action.name, transition.action.key, "feedback_only")
            self.game_memory.remember_feedback(transition.action.key)
        elif meaningful_change:
            self.game_memory.remember_effect(transition.action.name, transition.action.key, "changed_state")
        else:
            self.game_memory.remember_effect(transition.action.name, transition.action.key, "noop")
        if transition.reward_delta != 0:
            self.game_memory.remember_reward(transition.action.key, transition.reward_delta)
        self.game_memory.remember_transition_signature(transition.action.name, transition.action.key, transition.notes)
        if collectible_progress:
            self.game_memory.remember_collectible_progress(transition.action.key)
            self.game_memory.remember_effect(transition.action.name, transition.action.key, "collectible_progress")
        if transition.terminal and not transition.won:
            self.game_memory.remember_danger(transition.action.key)

    def _filter_useless_actions(self, observation: Observation, actions: list[Action]) -> list[Action]:
        filtered = []
        for action in actions:
            if action.name in self.game_memory.restart_like_action_names or action.key in self.game_memory.restart_like_action_keys:
                continue
            if action.name in self.game_memory.undo_like_action_names or action.key in self.game_memory.undo_like_action_keys:
                continue
            if self.game_memory.world_model.is_unsafe_action(action):
                continue
            if self.graph.action_is_probably_useless(observation.state_key, action):
                continue
            filtered.append(action)
        if filtered:
            return filtered
        non_meta = [
            action
            for action in actions
            if action.name not in self.game_memory.restart_like_action_names
            and action.key not in self.game_memory.restart_like_action_keys
            and action.name not in self.game_memory.undo_like_action_names
            and action.key not in self.game_memory.undo_like_action_keys
        ]
        return non_meta or actions

    def _learn_meta_action(self, transition: Transition) -> None:
        if transition.won or transition.reward_delta > 0:
            return
        if transition.from_state == transition.to_state:
            return
        previous_state = self.recent_states[-2] if len(self.recent_states) >= 2 else None
        if transition.terminal and not transition.won and transition.to_state in {
            getattr(self, "initial_state_key", None),
            previous_state,
        }:
            self.game_memory.remember_failure_revert(transition.action.name, transition.action.key)
        elif previous_state is not None and transition.to_state == previous_state:
            self.game_memory.remember_undo_like(transition.action.name, transition.action.key)
        elif transition.to_state == getattr(self, "initial_state_key", None):
            self.game_memory.remember_restart_like(transition.action.name, transition.action.key)

    def _learn_action_semantics(self, transition: Transition) -> None:
        previous_state = self.recent_states[-2] if len(self.recent_states) >= 2 else None
        before_player = transition.notes.get("semantic_previous_player_pos")
        after_player = transition.notes.get("semantic_player_pos")
        move_vector = None
        if isinstance(before_player, tuple) and isinstance(after_player, tuple):
            move_vector = (int(after_player[0]) - int(before_player[0]), int(after_player[1]) - int(before_player[1]))
        self.game_memory.learned_action_semantics.observe(
            transition,
            move_vector=move_vector,
            returned_previous=previous_state is not None and transition.to_state == previous_state,
            returned_initial=transition.to_state == getattr(self, "initial_state_key", None) and transition.from_state != transition.to_state,
        )

    def _fallback_action(self, observation: Observation, actions: list[Action]) -> Action:
        for action in actions:
            if action.name in self.game_memory.restart_like_action_names or action.key in self.game_memory.restart_like_action_keys:
                continue
            if action.name in self.game_memory.undo_like_action_names or action.key in self.game_memory.undo_like_action_keys:
                continue
            if self.game_memory.world_model.is_unsafe_action(action):
                continue
            if not self.graph.action_is_probably_useless(observation.state_key, action):
                return action
        for action in actions:
            if action.name in self.game_memory.restart_like_action_names or action.key in self.game_memory.restart_like_action_keys:
                continue
            if action.name in self.game_memory.undo_like_action_names or action.key in self.game_memory.undo_like_action_keys:
                continue
            return action
        return actions[0]
