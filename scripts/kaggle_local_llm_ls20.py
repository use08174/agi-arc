#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def bootstrap() -> Path:
    repo = Path(__file__).resolve().parent.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    cache_root = (
        Path("/kaggle/working/.arc_agi3_cache")
        if Path("/kaggle/working").exists()
        else Path("/tmp/.arc_agi3_cache")
    )
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg-cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg-cache"))
    return repo


def recommended_model_path(short_name: str) -> str:
    mappings = {
        "qwen-coder-1.5b": "/kaggle/input/qwen2-5-coder-1-5b-instruct",
        "qwen-coder-3b": "/kaggle/input/qwen2-5-coder-3b-instruct",
        "qwen-7b": "/kaggle/input/qwen2-5-7b-instruct",
    }
    return mappings.get(short_name, short_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARC with a local Kaggle LLM hook")
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument(
        "--mode",
        choices=["online", "competition", "offline", "normal"],
        default="online",
    )
    parser.add_argument(
        "--model-short-name",
        default="qwen-coder-1.5b",
        help="shortcut name or explicit local path",
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--explore-steps", type=int, default=24)
    parser.add_argument("--llm-start-step", type=int, default=8)
    parser.add_argument("--llm-step-interval", type=int, default=8)
    parser.add_argument("--llm-max-calls", type=int, default=12)
    parser.add_argument("--llm-max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--llm-thinking-mode",
        choices=["off", "brief", "full"],
        default="brief",
    )
    parser.add_argument("--llm-thinking-max-new-tokens", type=int, default=512)
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--llm-show-trace", action="store_true")
    parser.add_argument("--llm-show-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    repo = bootstrap()
    args = parse_args()
    model_path = recommended_model_path(args.model_short_name)
    if args.mode == "offline":
        from arc_agi3.runner.offline_eval import OfflineEvalConfig, run_offline_evaluation

        os.environ["ARC_AGI3_MAX_ACTIONS"] = str(args.max_steps)
        os.environ["ARC_AGI3_MAX_STEPS"] = str(args.max_steps)
        os.environ["ARC_AGI3_EXPLORE_STEPS"] = str(args.explore_steps)
        os.environ["ARC_AGI3_LLM_ENABLED"] = "1"
        os.environ["ARC_AGI3_LLM_PROVIDER"] = "transformers_local"
        os.environ["ARC_AGI3_LLM_MODEL_PATH"] = model_path
        os.environ["ARC_AGI3_LLM_DEVICE"] = args.llm_device
        os.environ["ARC_AGI3_LLM_START_STEP"] = str(args.llm_start_step)
        os.environ["ARC_AGI3_LLM_STEP_INTERVAL"] = str(args.llm_step_interval)
        os.environ["ARC_AGI3_LLM_MAX_CALLS"] = str(args.llm_max_calls)
        os.environ["ARC_AGI3_LLM_MAX_NEW_TOKENS"] = str(args.llm_max_new_tokens)
        os.environ["ARC_AGI3_LLM_THINKING_MODE"] = args.llm_thinking_mode
        os.environ["ARC_AGI3_LLM_THINKING_MAX_NEW_TOKENS"] = str(args.llm_thinking_max_new_tokens)
        os.environ["ARC_AGI3_LLM_SHOW_TRACE"] = "1" if args.llm_show_trace else "0"
        os.environ["ARC_AGI3_LLM_SHOW_PROMPT"] = "1" if args.llm_show_prompt else "0"
        run_offline_evaluation(
            OfflineEvalConfig(
                repo_root=repo,
                agent_src=repo / "scripts" / "kaggle_offline_eval_agent.py",
                run_game=args.game_id,
                description=f"agi-arc-offline-{args.game_id}",
            )
        )
        return

    from arc_agi3.runner.main import main as runner_main

    sys.argv = [
        "arc_agi3.runner.main",
        "--backend",
        "arcade",
        "--mode",
        args.mode,
        "--game-id",
        args.game_id,
        "--max-steps",
        str(args.max_steps),
        "--explore-steps",
        str(args.explore_steps),
        "--llm-enabled",
        "--llm-provider",
        "transformers_local",
        "--llm-model-path",
        model_path,
        "--llm-device",
        args.llm_device,
        "--llm-start-step",
        str(args.llm_start_step),
        "--llm-step-interval",
        str(args.llm_step_interval),
        "--llm-max-calls",
        str(args.llm_max_calls),
        "--llm-max-new-tokens",
        str(args.llm_max_new_tokens),
        "--llm-thinking-mode",
        args.llm_thinking_mode,
        "--llm-thinking-max-new-tokens",
        str(args.llm_thinking_max_new_tokens),
    ]
    if args.llm_show_trace:
        sys.argv.append("--llm-show-trace")
    if args.llm_show_prompt:
        sys.argv.append("--llm-show-prompt")
    runner_main()


if __name__ == "__main__":
    main()
