from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

from config import AppConfig


COMP_ROOT = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")


def clean_existing_pillow() -> None:
    candidates: list[Path] = []
    for base in site.getsitepackages():
        root = Path(base)
        candidates.extend(root.glob("PIL"))
        candidates.extend(root.glob("Pillow-*.dist-info"))
        candidates.extend(root.glob("pillow-*.dist-info"))

    user_site = Path(site.getusersitepackages())
    candidates.extend(user_site.glob("PIL"))
    candidates.extend(user_site.glob("Pillow-*.dist-info"))
    candidates.extend(user_site.glob("pillow-*.dist-info"))

    for path in candidates:
        if path.exists():
            print("Removing:", path)
            shutil.rmtree(path, ignore_errors=True)

    for name in list(sys.modules):
        if name == "PIL" or name.startswith("PIL."):
            sys.modules.pop(name, None)


def find_pillow_wheel() -> Path | None:
    wheel_dirs: list[Path] = []
    if custom_dir := os.getenv("PILLOW_WHEEL_DIR"):
        wheel_dirs.append(Path(custom_dir))
    wheel_dirs.extend(
        [
            Path.cwd() / "pillow-wheelhouse",
            Path("/kaggle/input/pillow-wheelhouse"),
            Path("/kaggle/working/pillow-wheelhouse"),
        ]
    )

    wheels: list[Path] = []
    for directory in wheel_dirs:
        if directory.exists():
            wheels.extend(sorted(directory.glob("Pillow-*.whl")))
            wheels.extend(sorted(directory.glob("pillow-*.whl")))
    if not wheels:
        return None

    preferred = [wheel for wheel in wheels if "11.3.0" in wheel.name]
    return preferred[0] if preferred else wheels[0]


def install_pillow_from_local_wheel() -> None:
    wheel = find_pillow_wheel()
    if wheel is None:
        print("WARNING: no local Pillow wheel found. Using preinstalled Pillow.")
    else:
        print("Installing Pillow from local wheel:", wheel)
        clean_existing_pillow()
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
                "-q",
            ]
        )
        for name in list(sys.modules):
            if name == "PIL" or name.startswith("PIL."):
                sys.modules.pop(name, None)

    import PIL

    print("Pillow OK:", PIL.__version__)
    print("Pillow file:", PIL.__file__)


def install_arc_agi_from_wheels(config: AppConfig) -> None:
    wheel_dir = COMP_ROOT / "arc_agi_3_wheels"
    if not config.in_kaggle or not wheel_dir.exists():
        return

    print("Installing arc-agi from competition wheels...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheel_dir),
            "arc-agi",
            "python-dotenv",
            "-q",
        ]
    )


def ensure_repo_src_path(config: AppConfig) -> None:
    for candidate in [
        config.repo_root / "src",
        Path.cwd() / "src",
        Path("/kaggle/working/agi-arc/src"),
    ]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def initialize_runtime(
    config: AppConfig,
    *,
    install_pillow: bool = False,
    install_arc: bool = True,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if install_pillow:
        install_pillow_from_local_wheel()
    if install_arc:
        install_arc_agi_from_wheels(config)
    ensure_repo_src_path(config)
    config.ensure_output_dirs()

    print("python:", sys.version.split()[0])
    try:
        import torch

        print("torch:", torch.__version__)
    except Exception as exc:
        print("torch import failed:", repr(exc))

    try:
        import transformers

        print("transformers:", transformers.__version__)
    except Exception as exc:
        print("transformers import failed:", repr(exc))

    print("repo root:", config.repo_root)
