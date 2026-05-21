from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
        task_dir = out_dir / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_json_path = task_dir / f"{task_id}.json"
        with task_json_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized_problem, handle)
        image_path = out_dir / f"{task_id}.png"
        image, _ = self._task2img.task_image(normalized_problem)
        image.save(image_path)
        explicit_binary = os.getenv("ARC_AGI3_ARCMDL_BIN")
        compiled_candidates = []
        if explicit_binary:
            compiled_candidates.append(Path(explicit_binary))
        compiled_candidates.extend(
            [
                self.vendor_root / "src" / "test",
                self.vendor_root / "src" / "batch",
                self.vendor_root / "src" / "arcprize",
                self.vendor_root / "src" / "arcathon",
            ]
        )
        compiled_binary = next((path for path in compiled_candidates if path.exists() and path.is_file()), None)
        result = {
            "available": True,
            "engine": "arcmdl",
            "task_image": str(image_path),
            "task_json": str(task_json_path),
            "compiled_binary": str(compiled_binary) if compiled_binary is not None else None,
            "mode": "image_only",
            "train_examples": len(normalized_problem.get("train", [])),
            "test_examples": len(normalized_problem.get("test", [])),
        }
        if compiled_binary is not None and os.getenv("ARC_AGI3_RUN_ARCMDL_CLI", "1") != "0":
            result.update(self._run_cli(compiled_binary, task_dir, task_id))
        return result

    def _run_cli(self, executable: Path, task_dir: Path, task_id: str) -> dict[str, Any]:
        if executable.name != "test":
            return {
                "mode": "compiled_unstructured",
                "cli_available": True,
                "cli_reason": f"found {executable.name}, but structured parsing is only wired for the test executable",
            }
        command = [
            str(executable),
            "-dir",
            f"{task_dir}/",
            "-tasks",
            task_id,
            "-learn",
            "-timeout_build",
            os.getenv("ARC_AGI3_ARCMDL_TIMEOUT_BUILD", "5"),
            "-timeout_prune",
            os.getenv("ARC_AGI3_ARCMDL_TIMEOUT_PRUNE", "1"),
            "-v",
            os.getenv("ARC_AGI3_ARCMDL_VERBOSITY", "1"),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(executable.parent),
                capture_output=True,
                text=True,
                timeout=int(os.getenv("ARC_AGI3_ARCMDL_TIMEOUT_TOTAL", "20")),
                check=False,
            )
        except Exception as exc:
            return {
                "mode": "compiled_failed",
                "cli_available": True,
                "cli_error": str(exc),
            }
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        log_path = self.artifact_dir / "arcmdl" / f"{task_id}.log"
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        parsed = self._parse_test_output(stdout)
        return {
            "mode": "compiled_cli",
            "cli_available": True,
            "cli_exit_code": int(completed.returncode),
            "cli_log_path": str(log_path),
            **parsed,
        }

    def _parse_test_output(self, stdout: str) -> dict[str, Any]:
        lines = stdout.splitlines()
        dl_lines = [line.strip() for line in lines if line.strip().startswith("DL ")]
        task_line = next((line.strip() for line in lines if "Checking task" in line), None)
        perf_lines = [
            line.strip()
            for line in lines
            if line.strip().startswith("acc-")
            or line.strip().startswith("bits-")
            or line.strip().startswith("runtime-")
        ]
        descriptive_model = self._extract_model_block(
            lines,
            marker="# Learned model (decriptive, before pruning):",
        )
        predictive_model = self._extract_model_block(
            lines,
            marker="# Learned model (predictive, after pruning):",
        )
        return {
            "task_line": task_line,
            "description_length_lines": dl_lines[:8],
            "descriptive_model": descriptive_model,
            "predictive_model": predictive_model,
            "performance_lines": perf_lines[:12],
            "md_model_present": bool(descriptive_model or predictive_model),
        }

    def _extract_model_block(self, lines: list[str], *, marker: str) -> list[str]:
        try:
            start = lines.index(marker) + 1
        except ValueError:
            return []
        block: list[str] = []
        for line in lines[start:]:
            stripped = line.rstrip()
            if not stripped:
                if block:
                    break
                continue
            if stripped.startswith("# ") or stripped.startswith("DL ") or stripped.startswith("## "):
                break
            block.append(stripped)
        return block[:24]
