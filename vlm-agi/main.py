from __future__ import annotations

import argparse
from config import AppConfig
from model import VLMManager
from runtime import initialize_runtime
from session import VLMArcRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refactored VLM ARC runner")
    parser.add_argument("--game-id", help="Override ARC game id")
    parser.add_argument("--max-steps", type=int, help="Override max episode steps")
    parser.add_argument(
        "--actions-per-call",
        type=int,
        help="Number of planned actions requested from the VLM per call",
    )
    parser.add_argument(
        "--close-at-end",
        action="store_true",
        help="Close the scorecard after the run",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable matplotlib/image display for frames",
    )
    parser.add_argument(
        "--install-pillow-fix",
        action="store_true",
        help="Reinstall Pillow from a local wheel before running",
    )
    parser.add_argument(
        "--skip-arc-install",
        action="store_true",
        help="Skip installing arc-agi wheels during startup",
    )
    parser.add_argument(
        "--keep-scorecard-open",
        action="store_true",
        help="Do not auto-close the scorecard at process exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.from_env()

    overrides = {}
    if args.game_id:
        overrides["game_id"] = args.game_id
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.actions_per_call is not None:
        overrides["actions_per_vlm_call"] = args.actions_per_call
    if overrides:
        config = config.with_overrides(**overrides)

    initialize_runtime(
        config,
        install_pillow=args.install_pillow_fix,
        install_arc=not args.skip_arc_install,
    )

    vlm = VLMManager(config)
    runner = VLMArcRunner(config, vlm)
    runner.start_arc_session(display_initial=not args.no_display)
    should_auto_close = (
        not args.keep_scorecard_open and config.arc_mode.lower() in {"online", "competition"}
    )

    try:
        runner.run_episode(
            max_steps=config.max_steps,
            close_at_end=args.close_at_end,
            display_after=not args.no_display,
        )
    finally:
        if should_auto_close and not runner.session.get("closed"):
            print("closing scorecard...")
            runner.close_scorecard()


if __name__ == "__main__":
    main()
