from __future__ import annotations

import argparse

from arc_agi3.core.logging_utils import get_logger
from arc_agi3.core.runtime_env import bootstrap_runtime_env

bootstrap_runtime_env()

try:
    from arc_agi import Arcade, OperationMode
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "arc-agi is not installed. Install it before using the scorecard CLI."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch an ARC scorecard")
    parser.add_argument("--scorecard-id", required=True)
    parser.add_argument(
        "--mode",
        choices=["online", "competition"],
        default="online",
    )
    parser.add_argument("--arc-base-url", default="https://three.arcprize.org")
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--recordings-dir", default="recordings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("arc_agi3.scorecard")
    arcade = Arcade(
        arc_base_url=args.arc_base_url,
        operation_mode=OperationMode(args.mode),
        environments_dir=args.environments_dir,
        recordings_dir=args.recordings_dir,
        logger=logger,
    )
    card = arcade.get_scorecard(args.scorecard_id)
    if card is None:
        print("scorecard not found")
        return

    print(f"card_id={card.card_id}")
    print(f"score={card.score}")
    print(f"environments={card.total_environments}")
    print(f"environments_completed={card.total_environments_completed}")
    print(f"levels_completed={card.total_levels_completed}")
    print(f"actions={card.total_actions}")
    print(f"url={args.arc_base_url.rstrip('/')}/scorecards/{card.card_id}")
    for env_score in card.environments:
        print(
            f"game={env_score.id} score={env_score.score} "
            f"levels_completed={env_score.levels_completed} actions={env_score.actions}"
        )


if __name__ == "__main__":
    main()
