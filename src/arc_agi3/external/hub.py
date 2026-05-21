from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from arc_agi3.core.types import Action, Frame
from arc_agi3.external.arcmdl_adapter import ArcMDLAdapter
from arc_agi3.external.compressarc_adapter import CompressARCAdapter
from arc_agi3.external.episode_task_export import EpisodeTaskExporter


class ExternalReasonerHub:
    """Connect vendored external ARC repos to the runtime."""

    def __init__(self, repo_root: Path | None = None) -> None:
        env_repo_root = os.getenv("ARC_AGI3_VENDOR_REPO") or os.getenv("AGI_ARC_REPO")
        self.repo_root = Path(env_repo_root) if env_repo_root else (repo_root or Path(__file__).resolve().parents[3])
        self.artifact_dir = self.repo_root / ".arc_agi3_external"
        self.exporter: EpisodeTaskExporter | None = None
        self.compressarc = CompressARCAdapter(self.repo_root)
        self.arcmdl = ArcMDLAdapter(self.repo_root, self.artifact_dir)
        self.use_compressarc = os.getenv("ARC_AGI3_USE_COMPRESSARC", "1") != "0"
        self.use_arcmdl = os.getenv("ARC_AGI3_USE_ARCMDL", "1") != "0"

    def reset(self, task_id: str, frame: Frame) -> dict[str, Any]:
        self.exporter = EpisodeTaskExporter(task_id=task_id)
        self.exporter.reset(frame)
        return self._analyze()

    def observe_transition(self, action: Action, frame: Frame) -> dict[str, Any]:
        if self.exporter is None:
            self.exporter = EpisodeTaskExporter(task_id="unknown")
            self.exporter.reset(frame)
        else:
            self.exporter.observe(action, frame)
        return self._analyze()

    def _analyze(self) -> dict[str, Any]:
        assert self.exporter is not None
        problem = self.exporter.build_problem()
        notes: dict[str, Any] = {
            "external_reasoner_summary": [],
            "external_reasoner_metadata": self.exporter.metadata(),
        }

        if self.use_compressarc:
            info = self.compressarc.analyze(self.exporter.task_id, problem)
            notes["external_compressarc"] = info
            if info.get("available"):
                notes["compressarc_n_colors"] = int(info.get("n_colors", 0) or 0)
                notes["compressarc_n_x"] = int(info.get("n_x", 0) or 0)
                notes["compressarc_n_y"] = int(info.get("n_y", 0) or 0)
                notes["external_reasoner_summary"].append(
                    "compressarc:"
                    f"colors={info.get('n_colors')} "
                    f"shape={info.get('n_x')}x{info.get('n_y')} "
                    f"same_size={int(bool(info.get('in_out_same_size')))}"
                )
                notes["compressarc_in_out_same_size"] = bool(info.get("in_out_same_size"))
                notes["compressarc_all_in_same_size"] = bool(info.get("all_in_same_size"))
                notes["compressarc_all_out_same_size"] = bool(info.get("all_out_same_size"))

        if self.use_arcmdl:
            info = self.arcmdl.analyze(self.exporter.task_id, problem)
            notes["external_arcmdl"] = info
            if info.get("available"):
                notes["arcmdl_train_examples"] = int(info.get("train_examples", 0) or 0)
                notes["external_reasoner_summary"].append(
                    "arcmdl:"
                    f"train={info.get('train_examples')} "
                    f"compiled={int(bool(info.get('compiled_binary')))}"
                )
                notes["arcmdl_compiled_available"] = bool(info.get("compiled_binary"))

        return notes
