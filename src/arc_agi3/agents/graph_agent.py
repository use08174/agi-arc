from __future__ import annotations

from arc_agi3.agents.base import ArcAgentRuntime
from arc_agi3.core.types import Action, Observation, Transition
from arc_agi3.envs.base import ArcEnvironment


class GraphSearchAgent(ArcAgentRuntime):
    """Explore-first agent scaffold for ARC-AGI-3 style tasks."""

    def run_episode(self, env: ArcEnvironment) -> tuple[bool, int]:
        self.reset_level()
        frame = env.reset()
        observation = self.hasher.observe(frame)
        self.graph.touch(observation.state_key, terminal=False)
        self.recent_states.append(observation.state_key)

        for step_idx in range(self.config.budget.max_steps_per_level):
            actions = env.valid_actions()
            action = self._choose_action(step_idx, observation, actions)
            result = env.step(action)
            next_observation = self.hasher.observe(result.frame, previous=observation.frame)

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
            self._learn_from_transition(transition)

            if result.done:
                return result.won, step_idx + 1

            observation = next_observation
            self.recent_states.append(observation.state_key)

        return False, self.config.budget.max_steps_per_level

    def _choose_action(
        self,
        step_idx: int,
        observation: Observation,
        actions: list[Action],
    ) -> Action:
        actions = self._filter_useless_actions(observation, actions)
        ranked_actions = self.llm.rank_actions(
            observation=observation,
            candidate_actions=actions,
            graph=self.graph,
            game_memory=self.game_memory,
            recent_states=list(self.recent_states),
            step_idx=step_idx,
        )
        actions = self.explorer.reorder_with_rankings(
            observation=observation,
            actions=actions,
            ranked_actions=ranked_actions,
            graph=self.graph,
            game_memory=self.game_memory,
            force_exploration=step_idx < self.config.budget.explore_phase_steps,
        )

        if step_idx >= self.config.budget.explore_phase_steps:
            plan = self.planner.build_plan(
                observation=observation,
                actions=actions,
                graph=self.graph,
                game_memory=self.game_memory,
                recent_states=set(self.recent_states),
            )
            if plan:
                return plan[0].action

        action = self.explorer.choose_action(
            observation=observation,
            actions=actions,
            graph=self.graph,
            game_memory=self.game_memory,
            recent_states=set(self.recent_states),
        )
        if action is None:
            return self._fallback_action(observation, actions)
        return action

    def _learn_from_transition(self, transition: Transition) -> None:
        likely_feedback_flash = bool(transition.notes.get("likely_feedback_flash", False))
        if likely_feedback_flash:
            self.game_memory.remember_effect(
                transition.action.name,
                transition.action.key,
                "feedback_only",
            )
            self.game_memory.remember_feedback(transition.action.key)
        elif transition.changed:
            self.game_memory.remember_effect(
                transition.action.name,
                transition.action.key,
                "changed_state",
            )
        else:
            self.game_memory.remember_effect(
                transition.action.name,
                transition.action.key,
                "noop",
            )
        if transition.reward_delta != 0:
            self.game_memory.remember_reward(
                transition.action.key,
                transition.reward_delta,
            )
        self.game_memory.remember_transition_signature(
            transition.action.name,
            transition.action.key,
            transition.notes,
        )
        if bool(transition.notes.get("collectible_progress", False)):
            self.game_memory.remember_collectible_progress(transition.action.key)
            self.game_memory.remember_effect(
                transition.action.name,
                transition.action.key,
                "collectible_progress",
            )
        if transition.terminal and not transition.won:
            self.game_memory.remember_danger(transition.action.key)
        if likely_feedback_flash and transition.reward_delta <= 0:
            self.game_memory.remember_danger(transition.action.key)

    def _filter_useless_actions(
        self,
        observation: Observation,
        actions: list[Action],
    ) -> list[Action]:
        filtered = [
            action
            for action in actions
            if not self.graph.action_is_probably_useless(observation.state_key, action)
        ]
        return filtered or actions

    def _fallback_action(
        self,
        observation: Observation,
        actions: list[Action],
    ) -> Action:
        for action in actions:
            if not self.graph.action_is_probably_useless(observation.state_key, action):
                return action
        return actions[0]
