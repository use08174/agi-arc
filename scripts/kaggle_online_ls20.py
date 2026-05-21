#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_src_on_path() -> None:
    src = repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def kaggle_cache_root() -> Path:
    if Path("/kaggle/working").exists():
        root = Path("/kaggle/working/.arc_agi3_cache")
    else:
        root = Path("/tmp/.arc_agi3_cache")
    root.mkdir(parents=True, exist_ok=True)
    (root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (root / "xdg-cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "xdg-cache"))


def maybe_install_arc_agi(args: argparse.Namespace) -> None:
    if args.install_mode == "skip":
        return

    try:
        import arc_agi  # noqa: F401
        return
    except Exception:
        pass

    python = sys.executable
    if args.install_mode == "online":
        subprocess.check_call([python, "-m", "pip", "install", "arc-agi"])
        return

    if args.install_mode == "wheel-dir":
        if not args.wheel_dir:
            raise SystemExit("--wheel-dir is required when --install-mode wheel-dir")
        subprocess.check_call(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={args.wheel_dir}",
                "arc-agi",
            ]
        )
        return

    raise SystemExit(f"unsupported install mode: {args.install_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle-friendly ARC online runner")
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument(
        "--mode",
        choices=["online", "competition", "offline", "normal"],
        default="online",
    )
    parser.add_argument(
        "--install-mode",
        choices=["skip", "online", "wheel-dir"],
        default="skip",
    )
    parser.add_argument("--wheel-dir", default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("ARC_AGI3_MAX_STEPS", "128")),
    )
    parser.add_argument(
        "--explore-steps",
        type=int,
        default=int(os.getenv("ARC_AGI3_EXPLORE_STEPS", "24")),
    )
    parser.add_argument("--list-games", action="store_true")
    parser.add_argument("--save-recording", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle_cache_root()
    ensure_src_on_path()
    os.environ.setdefault("AGI_ARC_REPO", str(repo_root()))
    os.environ.setdefault("ARC_AGI3_USE_COMPRESSARC", "1")
    os.environ.setdefault("ARC_AGI3_USE_ARCMDL", "0")
    os.environ.setdefault("ARC_AGI3_RUN_ARCMDL_CLI", "0")
    maybe_install_arc_agi(args)

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
    ]
    if args.list_games:
        sys.argv.append("--list-games")
    if args.save_recording:
        sys.argv.append("--save-recording")
    runner_main()


if __name__ == "__main__":
    main()
