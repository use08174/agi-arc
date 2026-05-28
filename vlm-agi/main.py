from __future__ import annotations

import argparse
import json

from config import AppConfig
from model import VLMManager
from runtime import initialize_runtime
from scene_probe import run_scene_probe_sequence
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
        "--scene-probe",
        action="store_true",
        help="Run the scene-understanding probe loop instead of the plain episode loop",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=5,
        help="Number of probe steps when --scene-probe is enabled",
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

    if args.scene_probe:
        results = run_scene_probe_sequence(
            runner,
            steps=args.probe_steps,
            display_after=not args.no_display,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return

    logs = runner.run_episode(
        max_steps=config.max_steps,
        close_at_end=args.close_at_end,
        display_after=not args.no_display,
    )
    print(json.dumps(logs[-3:], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
