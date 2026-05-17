#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if Path("/kaggle/working").exists():
        cache_root = Path("/kaggle/working/.arc_agi3_cache")
    else:
        cache_root = Path("/tmp/.arc_agi3_cache")
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg-cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg-cache"))


def main() -> None:
    bootstrap()
    from arc_agi3.runner.scorecard import main as scorecard_main

    scorecard_main()


if __name__ == "__main__":
    main()
