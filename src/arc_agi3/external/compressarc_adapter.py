from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class CompressARCAdapter:
    """Thin wrapper around the vendored CompressARC preprocessing code."""

    def __init__(self, repo_root: Path) -> None:
        self.vendor_root = repo_root / "vendor" / "CompressARC"
        self.available = self.vendor_root.exists()
        self._preprocessing: ModuleType | None = None

    def _load_module(self, module_name: str, relative_path: str) -> ModuleType:
        if module_name in sys.modules:
            return sys.modules[module_name]
        module_path = self.vendor_root / relative_path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module at {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_loaded(self) -> None:
        if self._preprocessing is not None:
            return
        sys.path.insert(0, str(self.vendor_root))
        self._load_module("multitensor_systems", "multitensor_systems.py")
        self._preprocessing = self._load_module("compressarc_preprocessing", "preprocessing.py")

    def analyze(self, task_id: str, problem: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "engine": "compressarc"}
        self._ensure_loaded()
        assert self._preprocessing is not None
        task = self._preprocessing.Task(task_id, problem, None)
        return {
            "available": True,
            "engine": "compressarc",
            "n_train": int(task.n_train),
            "n_test": int(task.n_test),
            "n_examples": int(task.n_examples),
            "n_colors": int(task.n_colors),
            "n_x": int(task.n_x),
            "n_y": int(task.n_y),
            "in_out_same_size": bool(task.in_out_same_size),
            "all_in_same_size": bool(task.all_in_same_size),
            "all_out_same_size": bool(task.all_out_same_size),
        }
