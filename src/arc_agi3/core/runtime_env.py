from __future__ import annotations

import os
from pathlib import Path


def bootstrap_runtime_env() -> None:
    """Set cache/config paths to writable locations.

    This is especially useful on Kaggle and other notebook sandboxes where
    default home-directory cache locations are read-only or missing.
    """

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle").exists():
        root = Path("/kaggle/working/.arc_agi3_cache")
    else:
        root = Path("/private/tmp/arc_agi3_cache")

    root.mkdir(parents=True, exist_ok=True)
    mpl = root / "mplconfig"
    xdg = root / "xdg-cache"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg))
