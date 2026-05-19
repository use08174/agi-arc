from __future__ import annotations

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
        self.recent_actions.append(action.name)
        if discovered_new_state:
            self.steps_since_new_state = 0
        else:
            self.steps_since_new_state += 1
        semantic_progress = bool(
            reward_delta > 0
            or next_observation.notes.get("collectible_progress", False)
            or next_observation.notes.get("anchor_patch_changes", [])
            or (experiment_outcome is not None and experiment_outcome.status == "confirmed")
        )
        if semantic_progress:
            self.steps_since_semantic_progress = 0
        else:
            self.steps_since_semantic_progress += 1
        self.recent_states.append(next_observation.state_key)
        return next_observation

    def _choose_action(self, step_idx: int, observation: Observation, actions: list[Action]) -> Action:
        actions = self._filter_useless_actions(observation, actions)
        force_exploration = self._should_force_exploration(step_idx, observation)
        actions = self._prefer_unseen_actions(observation, actions, force_exploration)
        self._activate_experiment_if_idle(observation, actions, force_exploration)
        experiment_plan = self.planner.build_experiment_plan(
            experiment=self.game_memory.experiments.active,
            actions=actions,
            game_memory=self.game_memory,
        )
        plan_outcome = self.game_memory.experiments.note_plan_result(plannable=bool(experiment_plan))
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
            )
            if counterfactual is not None:
                return self._trace_decision(step_idx, observation, counterfactual, "counterfactual", "breaking repetitive low-progress action loop")
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

    def _activate_experiment_if_idle(self, observation: Observation, actions: list[Action], force_exploration: bool) -> None:
        if self.game_memory.experiments.active is not None or not actions:
            return
        current_node = self.graph.nodes.get(observation.state_key)
        seen_action_keys = set(current_node.outgoing) if current_node is not None else set()
        proposals = self.game_memory.experiments.available(
            self.game_memory.world_model,
            actions,
            seen_action_keys,
        )
        if not proposals:
            return
        selected = self._select_experiment(proposals, force_exploration)
        if selected is None:
            return
        self.game_memory.experiments.activate(selected)

    def _select_experiment(self, proposals, force_exploration: bool):
        world = self.game_memory.world_model
        known_axes = {
            axis
            for dx, dy in world.action_move_vectors.values()
            for axis in (
                "horizontal" if dx != 0 else "",
                "vertical" if dy != 0 else "",
            )
            if axis
        }

        def priority(proposal) -> tuple:
            kind = proposal.kind
            if kind == "discover_axis":
                axis = str((proposal.target or {}).get("axis", ""))
                return (0, axis in known_axes, proposal.key)
            if kind == "probe_action":
                return (1 if force_exploration else 2, proposal.key)
            if kind == "inspect_affordance":
                return (3, -float(proposal.confidence or 0.0), proposal.key)
            if kind in {"collect_item", "activate_button", "go_to_goal"}:
                return (4, proposal.key)
            if kind == "inspect_relation":
                return (5, proposal.key)
            return (6, proposal.key)

        return sorted(proposals, key=priority)[0] if proposals else None

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
            if action.key in self.game_memory.dangerous_action_keys:
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
        ranked = []
        recent = set(self.recent_actions)
        for index, action in enumerate(actions):
            if action.name in self.game_memory.restart_like_action_names or action.key in self.game_memory.restart_like_action_keys:
                continue
            if action.name in self.game_memory.undo_like_action_names or action.key in self.game_memory.undo_like_action_keys:
                continue
            if action.key in self.game_memory.dangerous_action_keys:
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
