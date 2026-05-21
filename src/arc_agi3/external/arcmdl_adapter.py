from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class ArcMDLAdapter:
    """Thin wrapper around the vendored ARC-MDL Python utility code."""

    def __init__(self, repo_root: Path, artifact_dir: Path) -> None:
        self.vendor_root = repo_root / "vendor" / "ARC-MDL"
        self.available = self.vendor_root.exists()
        self.artifact_dir = artifact_dir
        self._task2img: ModuleType | None = None

    def _load_module(self) -> None:
        if self._task2img is not None:
            return
        module_path = self.vendor_root / "src" / "task2img.py"
        spec = importlib.util.spec_from_file_location("arcmdl_task2img", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module at {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["arcmdl_task2img"] = module
        spec.loader.exec_module(module)
        self._task2img = module

    def analyze(self, task_id: str, problem: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "engine": "arcmdl"}
        self._load_module()
        assert self._task2img is not None
        normalized_problem = {
            "train": list(problem.get("train", [])),
            "test": [
                item if "output" in item else {**item, "output": item.get("input", [])}
                for item in problem.get("test", [])
            ],
        }
        out_dir = self.artifact_dir / "arcmdl"
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / f"{task_id}.png"
        image, _ = self._task2img.task_image(normalized_problem)
        image.save(image_path)
        compiled_candidates = [
            self.vendor_root / "src" / "batch",
            self.vendor_root / "src" / "arcprize",
            self.vendor_root / "src" / "arcathon",
        ]
        compiled_binary = next((str(path) for path in compiled_candidates if path.exists()), None)
        return {
            "available": True,
            "engine": "arcmdl",
            "task_image": str(image_path),
            "compiled_binary": compiled_binary,
            "train_examples": len(normalized_problem.get("train", [])),
            "test_examples": len(normalized_problem.get("test", [])),
        }
