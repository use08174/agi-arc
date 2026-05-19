from __future__ import annotations

import argparse
from pathlib import Path

from arc_agi3.core.runtime_env import bootstrap_runtime_env

bootstrap_runtime_env()

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig, RuntimeConfig
from arc_agi3.core.logging_utils import get_logger
from arc_agi3.envs.factory import build_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 starter runner")
    parser.add_argument("--backend", choices=["mock", "arcade"], default="mock")
    parser.add_argument("--game-id", default="ms00")
    parser.add_argument(
        "--mode",
        choices=["offline", "online", "competition", "normal"],
        default="offline",
    )
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--explore-steps", type=int, default=24)
    parser.add_argument("--novelty-patience-steps", type=int, default=6)
    parser.add_argument("--revisit-limit", type=int, default=3)
    parser.add_argument("--list-games", action="store_true")
    parser.add_argument("--save-recording", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("arc_agi3.runner")
    agent_config = AgentConfig()
    agent_config.budget.max_steps_per_level = args.max_steps
    agent_config.budget.explore_phase_steps = args.explore_steps
    agent_config.budget.novelty_patience_steps = args.novelty_patience_steps
    agent_config.budget.revisit_limit = args.revisit_limit

    runtime = RuntimeConfig(
        backend=args.backend,
        game_id=args.game_id,
        mode=args.mode,
        render_mode=args.render_mode,
        environments_dir=Path(args.environments_dir),
        recordings_dir=Path(args.recordings_dir),
        save_recording=args.save_recording,
    )

    env = build_environment(runtime, agent_config)
    try:
        if args.list_games:
            if hasattr(env, "list_available_games"):
                for game_id in env.list_available_games():
                    print(game_id)
            else:
                print("mock")
            return

        agent = GraphSearchAgent(config=agent_config)
        won, steps = agent.run_episode(env)
        print(f"backend={args.backend} game_id={args.game_id} won={won} steps={steps}")
        print(f"max_steps_budget={agent_config.budget.max_steps_per_level}")
        print(f"episode_end_reason={agent.last_episode_end_reason}")
        print(f"final_status={agent.last_episode_final_status}")
        final_tool_state = agent.last_episode_final_info.get("tool_state")
        if final_tool_state is not None:
            print(f"final_tool_state={final_tool_state}")
        print("promising_actions=", sorted(agent.game_memory.promising_actions))
        print("known_states=", len(agent.graph.nodes))
        if agent.decision_traces:
            print("recent_decisions=")
            for trace in agent.decision_traces[-12:]:
                print(
                    f"- step={trace.step_idx} state={trace.state_key} action={trace.action.key} "
                    f"source={trace.source} reason={trace.reason}"
                )
    finally:
        try:
            env.close()
            if hasattr(env, "last_closed_scorecard") and env.last_closed_scorecard is not None:
                card = env.last_closed_scorecard
                print(f"scorecard_id={card.card_id}")
                print(f"score={card.score}")
                if hasattr(card, "total_environments_completed"):
                    print(
                        "environments_completed="
                        f"{card.total_environments_completed}/{card.total_environments}"
                    )
                    print(f"levels_completed={card.total_levels_completed}")
                    print(f"actions={card.total_actions}")
                if hasattr(env, "scorecard_url"):
                    print(f"scorecard_url={env.scorecard_url()}")
                if hasattr(env, "has_online_replays") and env.has_online_replays():
                    print("replays_available=yes")
                elif hasattr(env, "has_online_replays"):
                    print("replays_available=no")
        except Exception:
            logger.exception("Failed to close environment cleanly")


if __name__ == "__main__":
    main()
