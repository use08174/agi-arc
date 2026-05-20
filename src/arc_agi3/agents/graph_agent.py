from __future__ import annotations

from collections import Counter

from arc_agi3.agents.base import ArcAgentRuntime
from arc_agi3.agents.env_understanding_agent import EnvUnderstandingAgent
from arc_agi3.core.types import Action, DecisionTrace, Frame, GameStatus, Observation, Transition
from arc_agi3.envs.base import ArcEnvironment
from arc_agi3.planning.safety_shield import SafetyShield


class GraphSearchAgent(ArcAgentRuntime):
    """Explore-first agent scaffold with semantic world-model updates."""

    def begin_external_episode(self, frame: Frame) -> None:
        self.last_episode_end_reason = "not_started"
        self.last_episode_final_status = "UNKNOWN"
        self.last_episode_final_info = {}
        self.reset_level()
        self.safety_shield = SafetyShield()
        self.understanding_agent = EnvUnderstandingAgent()
        observation = self.hasher.observe(frame)
        self.game_memory.world_model.update_from_observation(observation.notes)
        self.graph.touch(observation.state_key, terminal=False)
        self.initial_state_key = observation.state_key
        self.recent_states.append(observation.state_key)
        self._external_observation = observation
        self._external_pending_action: Action | None = None
        self._external_step_idx = 0

    def choose_external_action(self, frame: Frame, actions: list[Action]) -> Action:
        if getattr(self, "_external_observation", None) is None:
            self.begin_external_episode(frame)
        elif getattr(self, "_external_pending_action", None) is not None:
            self.observe_external_frame(frame)

        observation = self._external_observation
        action = self._choose_action(self._external_step_idx, observation, actions)
        action = self.safety_shield.validate_or_replace(action, observation, actions, self.game_memory)
        self._print_action_trace(self._external_step_idx, observation, action)
        self._external_pending_action = action
        self._external_step_idx += 1
        return action

    def observe_external_frame(self, frame: Frame) -> None:
        observation = getattr(self, "_external_observation", None)
        action = getattr(self, "_external_pending_action", None)
        if observation is None or action is None:
            return
        previous_levels = int(observation.frame.info.get("levels_completed", 0) or 0)
        next_levels = int(frame.info.get("levels_completed", 0) or 0)
        reward_delta = float(next_levels - previous_levels)
        done = frame.status in {GameStatus.WIN, GameStatus.GAME_OVER}
        won = frame.status == GameStatus.WIN
        next_observation = self._integrate_transition(
            observation=observation,
            action=action,
            frame=frame,
            reward_delta=reward_delta,
            done=done,
            won=won,
        )
        self._external_observation = next_observation
        self._external_pending_action = None

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
            self._print_action_trace(step_idx, observation, action)
            result = env.step(action)
            next_observation = self._integrate_transition(
                observation=observation,
                action=action,
                frame=result.frame,
                reward_delta=result.reward_delta,
                done=result.done,
                won=result.won,
            )
            if result.done:
                self.last_episode_end_reason = "environment_done"
                self.last_episode_final_status = result.frame.status.value
                self.last_episode_final_info = dict(result.frame.info)
                return result.won, step_idx + 1
            observation = next_observation
        self.last_episode_end_reason = "step_budget_exhausted"
        self.last_episode_final_status = observation.frame.status.value
        self.last_episode_final_info = dict(observation.frame.info)
        return False, self.config.budget.max_steps_per_level

    def _integrate_transition(
        self,
        observation: Observation,
        action: Action,
        frame: Frame,
        reward_delta: float,
        done: bool,
        won: bool,
    ) -> Observation:
        before_notes = dict(observation.notes)
        prior_action_key = self.recent_action_keys[-1] if self.recent_action_keys else None
        next_observation = self.hasher.observe(frame, previous=observation.frame)
        discovered_new_state = next_observation.state_key not in self.graph.nodes
        self.game_memory.world_model.update_from_observation(next_observation.notes)
        transition = Transition(
            from_state=observation.state_key,
            action=action,
            to_state=next_observation.state_key,
            changed=next_observation.changed,
            reward_delta=reward_delta,
            terminal=done,
            won=won,
            notes=next_observation.notes,
        )
        self.graph.record(transition)
        self.game_memory.world_model.learn_transition(
            action=action,
            before_notes=before_notes,
            after_notes=next_observation.notes,
            terminal_loss=done and not won,
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
        if discovered_new_state:
            self.steps_since_new_state = 0
        else:
            self.steps_since_new_state += 1
        progress_signal = self.game_memory.progress_model.score_transition(
            transition=transition,
            after_notes=next_observation.notes,
            discovered_new_state=discovered_new_state,
            experiment_outcome=experiment_outcome,
        )
        next_observation.notes.update(progress_signal.to_notes())
        transition.notes.update(progress_signal.to_notes())
        semantic_progress = progress_signal.is_progress
        previous_state = self.recent_states[-2] if len(self.recent_states) >= 2 else None
        signature = self.game_memory.action_effects.observe(
            action=action,
            transition=transition,
            notes=transition.notes,
            progress_score=progress_signal.score,
            returned_previous=previous_state is not None and transition.to_state == previous_state,
            returned_initial=transition.to_state == getattr(self, "initial_state_key", None) and transition.from_state != transition.to_state,
            previous_action_key=prior_action_key,
            latent_state=dict(self.game_memory.world_model.latent_state_candidates),
        )
        self.recent_actions.append(action.name)
        self.recent_action_keys.append(action.key)
        self.recent_action_families.append(
            self.game_memory.action_family(
                action.name,
                action.key,
                previous_action_key=prior_action_key,
                region_bias=str(transition.notes.get("region_bias", "playfield")),
            )
        )
        self.recent_effect_transforms.append(signature.transform_kind)
        self.recent_progress_scores.append(progress_signal.score)
        if semantic_progress:
            self.steps_since_semantic_progress = 0
        else:
            self.steps_since_semantic_progress += 1
        if action.payload:
            if semantic_progress:
                self.click_no_progress_counts.pop(action.key, None)
            else:
                self.click_no_progress_counts[action.key] = self.click_no_progress_counts.get(action.key, 0) + 1
        self._learn_macro_semantics(action, transition, progress_signal.score)
        self.recent_states.append(next_observation.state_key)
        return next_observation

    def _choose_action(self, step_idx: int, observation: Observation, actions: list[Action]) -> Action:
        original_actions = list(actions)
        actions = self._filter_useless_actions(observation, actions)
        force_exploration = self._should_force_exploration(step_idx, observation)
        actions = self._prefer_unseen_actions(observation, actions, force_exploration)
        actions = self._reprioritize_click_actions(actions)
        self._activate_experiment_if_idle(observation, actions, force_exploration)
        experiment_step = self.experiment_runner.build_step_from_session(
            session=self.game_memory.experiments.active_session,
            actions=actions,
            game_memory=self.game_memory,
        )
        if experiment_step is not None:
            self.game_memory.experiments.note_plan_result(plannable=True)
            self.game_memory.experiments.note_execution(experiment_step.action)
            return self._trace_decision(step_idx, observation, experiment_step.action, "experiment", experiment_step.reason)
        plan_outcome = self.game_memory.experiments.note_plan_result(
            plannable=self.game_memory.experiments.active is None
        )
        if plan_outcome is not None:
            self.game_memory.world_model.hypothesis_library.apply_experiment_result(
                kind=plan_outcome.proposal.kind,
                status=plan_outcome.status,
                evidence=plan_outcome.evidence,
            )
        if self._should_force_counterfactual_exploration():
            counterfactual = self.explorer.choose_counterfactual_action(
                observation=observation,
                actions=actions,
                graph=self.graph,
                game_memory=self.game_memory,
                recent_action_names=list(self.recent_actions),
                recent_action_keys=list(self.recent_action_keys),
            )
            if counterfactual is not None:
                return self._trace_decision(step_idx, observation, counterfactual, "counterfactual", "breaking repetitive low-progress action loop")
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
            fallback = self._fallback_action(observation, actions or original_actions)
            return self._trace_decision(step_idx, observation, fallback, "fallback", "no ranked frontier action")
        return self._trace_decision(step_idx, observation, action, "explorer", "frontier selection")

    def _prefer_unseen_actions(self, observation: Observation, actions: list[Action], force_exploration: bool) -> list[Action]:
        if not actions:
            return actions
        current_node = self.graph.nodes.get(observation.state_key)
        if current_node is None or not current_node.outgoing:
            return actions
        revisit_count = sum(1 for state in self.recent_states if state == observation.state_key)
        if not force_exploration and revisit_count < 2:
            return actions
        unseen = [action for action in actions if self.graph.seen_successor(observation.state_key, action) is None]
        return unseen or actions

    def _reprioritize_click_actions(self, actions: list[Action]) -> list[Action]:
        if not actions:
            return actions
        recent_key_counts = Counter(self.recent_action_keys)

        def sort_key(action: Action) -> tuple:
            if not action.payload:
                return (0, 0, 0, action.key)
            return (
                1,
                self.click_no_progress_counts.get(action.key, 0),
                recent_key_counts.get(action.key, 0),
                action.key,
            )

        return sorted(actions, key=sort_key)

    def _activate_experiment_if_idle(self, observation: Observation, actions: list[Action], force_exploration: bool) -> None:
        if self.game_memory.experiments.active is not None or not actions:
            return
        current_node = self.graph.nodes.get(observation.state_key)
        seen_action_keys = set(current_node.outgoing) if current_node is not None else set()
        proposals = self.game_memory.experiments.available(
            self.game_memory.world_model,
            actions,
            seen_action_keys,
            family_for=lambda action: self.game_memory.action_family(
                action.name,
                action.key,
                previous_action_key=self.recent_action_keys[-1] if self.recent_action_keys else None,
            ),
        )
        if not proposals:
            return
        selected = self.experiment_policy.select(
            proposals,
            self.game_memory.world_model,
            force_exploration,
            mode_action_keys=set(self.game_memory.experiments.mode_action_keys),
        )
        if selected is None:
            return
        self.game_memory.experiments.activate(selected)

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

    def _print_action_trace(self, step_idx: int, observation: Observation, action: Action) -> None:
        source = self.decision_traces[-1].source if self.decision_traces else "unknown"
        print(
            f"action_step={step_idx} state={observation.state_key} "
            f"action={action.key} source={source}"
        )

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
        progress_score = float(notes.get("progress_score", 0.0) or 0.0)
        hud_only_change = changed_hud_cells > 0 and changed_playfield_cells == 0 and transition.reward_delta <= 0 and not collectible_progress
        meaningful_change = changed_playfield_cells > 0 or transition.reward_delta > 0 or collectible_progress or semantic_player_moved
        if likely_feedback_flash or hud_only_change:
            self.game_memory.remember_effect(transition.action.name, transition.action.key, "feedback_only")
            self.game_memory.remember_feedback(transition.action.key)
        elif progress_score >= 0.45:
            self.game_memory.remember_effect(transition.action.name, transition.action.key, "progress")
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
        retried = []
        family_cooled = []
        for action in actions:
            if action.name in self.game_memory.restart_like_action_names or action.key in self.game_memory.restart_like_action_keys:
                continue
            if action.name in self.game_memory.undo_like_action_names or action.key in self.game_memory.undo_like_action_keys:
                continue
            if action.key in self.game_memory.dangerous_action_keys:
                continue
            if action.payload and self.click_no_progress_counts.get(action.key, 0) >= 2:
                continue
            if self.game_memory.world_model.is_unsafe_action(action):
                continue
            if self.graph.action_is_probably_useless(observation.state_key, action):
                continue
            if self._would_repeat_recent_action_pattern(action):
                continue
            if self._would_repeat_recent_family_pattern(action) or self._family_is_on_cooldown(action):
                family_cooled.append(action)
                continue
            if self.graph.action_was_tried(observation.state_key, action):
                retried.append(action)
                continue
            filtered.append(action)
        if filtered:
            return filtered
        if family_cooled:
            return family_cooled
        non_meta = [
            action
            for action in actions
            if action.name not in self.game_memory.restart_like_action_names
            and action.key not in self.game_memory.restart_like_action_keys
            and action.name not in self.game_memory.undo_like_action_names
            and action.key not in self.game_memory.undo_like_action_keys
        ]
        unseen_non_meta = [
            action
            for action in non_meta
            if not self.graph.action_was_tried(observation.state_key, action)
        ]
        if unseen_non_meta:
            return unseen_non_meta
        # If every action in this exact state has already been tried, fall back to the
        # retried set so higher-level ranking/counterfactual logic can still choose
        # the least-recent or least-looping option instead of hard-locking to one action.
        return retried or non_meta or []

    def _would_repeat_recent_action_pattern(self, action: Action) -> bool:
        if self.steps_since_semantic_progress <= 0:
            return False
        sequence = list(self.recent_action_keys) + [action.key]
        for pattern_len in (2, 3, 4):
            window = pattern_len * 2
            if len(sequence) < window:
                continue
            if sequence[-window:-pattern_len] == sequence[-pattern_len:]:
                return True
        return False

    def _would_repeat_recent_family_pattern(self, action: Action) -> bool:
        if self.steps_since_semantic_progress <= 1:
            return False
        family = self.game_memory.action_family(
            action.name,
            action.key,
            previous_action_key=self.recent_action_keys[-1] if self.recent_action_keys else None,
        )
        if family in {"movement", "direct_click"}:
            return False
        sequence = list(self.recent_action_families) + [family]
        for pattern_len in (2, 3):
            window = pattern_len * 2
            if len(sequence) < window:
                continue
            if sequence[-window:-pattern_len] == sequence[-pattern_len:]:
                return True
        return False

    def _family_is_on_cooldown(self, action: Action) -> bool:
        if self.steps_since_semantic_progress < 2 or len(self.recent_action_families) < 4:
            return False
        family = self.game_memory.action_family(
            action.name,
            action.key,
            previous_action_key=self.recent_action_keys[-1] if self.recent_action_keys else None,
        )
        if family in {"movement", "direct_click", "restart", "undo"}:
            return False
        recent_families = list(self.recent_action_families)[-5:]
        recent_progress = list(self.recent_progress_scores)[-5:]
        recent_transforms = list(self.recent_effect_transforms)[-5:]
        low_progress_same_family = [
            idx
            for idx, recent_family in enumerate(recent_families)
            if recent_family == family and recent_progress[idx] < 0.25
        ]
        if len(low_progress_same_family) < 3:
            return False
        transform_matches = {
            recent_transforms[idx]
            for idx in low_progress_same_family
        }
        return len(transform_matches) <= 2

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

    def _learn_macro_semantics(self, action: Action, transition: Transition, progress_score: float) -> None:
        previous = self.previous_action_context
        current_changed = bool(transition.changed)
        current_productive = progress_score >= 0.45 or transition.reward_delta > 0 or transition.won
        semantic_player_moved = bool(transition.notes.get("semantic_player_moved", False))
        if previous is not None:
            previous_key = str(previous.get("action_key", ""))
            previous_progress = float(previous.get("progress_score", 0.0) or 0.0)
            previous_changed = bool(previous.get("changed", False))
            if previous_key:
                mode_setting = (
                    previous_progress < 0.2
                    and not previous_changed
                    and current_productive
                    and not semantic_player_moved
                )
                productive_pair = (not semantic_player_moved) and (
                    current_productive or (
                        current_changed
                        and str(transition.notes.get("interaction_hint", "unknown")) not in {"unknown", "hud_or_counter_update"}
                    )
                )
                if mode_setting or productive_pair:
                    self.game_memory.experiments.remember_macro_effect(
                        previous_key,
                        action.key,
                        mode_setting=mode_setting,
                        productive=productive_pair,
                    )
        self.previous_action_context = {
            "action_key": action.key,
            "progress_score": progress_score,
            "changed": current_changed,
        }

    def _fallback_action(self, observation: Observation, actions: list[Action]) -> Action:
        ranked = []
        recent = set(self.recent_actions)
        for index, action in enumerate(actions):
            if action.name in self.game_memory.restart_like_action_names or action.key in self.game_memory.restart_like_action_keys:
                continue
            if action.name in self.game_memory.undo_like_action_names or action.key in self.game_memory.undo_like_action_keys:
                continue
            if action.key in self.game_memory.dangerous_action_keys:
                continue
            if action.payload and self.click_no_progress_counts.get(action.key, 0) >= 2:
                continue
            if self.game_memory.world_model.is_unsafe_action(action):
                continue
            meaning = self.game_memory.learned_action_semantics.meaning_for(action.name)
            ranked.append(
                (
                    self.graph.action_is_probably_useless(observation.state_key, action),
                    self.game_memory.world_model.is_blocked_action(action),
                    action.name in recent,
                    meaning.uses,
                    index,
                    action,
                )
            )
        if ranked:
            ranked.sort(key=lambda item: item[:-1])
            return ranked[0][-1]
        return actions[0]
