from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    game_id: str
    arc_mode: str
    arc_api_key: str
    arc_base_url: str
    environments_dir: str
    recordings_dir: str
    log_path: str
    summary_path: str
    max_steps: int
    max_new_tokens: int
    image_scale: int
    draw_grid: bool
    print_vlm_output: bool
    allow_download: bool
    force_reload_model: bool
    save_recording: bool
    actions_per_vlm_call: int
    stop_sequence_on_no_change: bool
    stop_sequence_on_level_change: bool
    stop_sequence_on_actions_change: bool
    action6_max_candidates: int
    action6_grid_points_per_axis: int
    max_actions_per_vlm_call: int
    adaptive_action_planning: bool
    repo_root: Path
    in_kaggle: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        in_kaggle = Path("/kaggle/input").exists()
        repo_root = Path(
            os.getenv(
                "AGI_ARC_REPO",
                "/kaggle/working/agi-arc" if in_kaggle else Path.cwd(),
            )
        )
        return cls(
            model_path=os.getenv(
                "LOCAL_VLM_MODEL_PATH",
                "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
            ),
            game_id=os.getenv("ARC_GAME_ID", "su15"),
            arc_mode=os.getenv("ARC_MODE", "online"),
            arc_api_key=os.getenv("ARC_API_KEY", ""),
            arc_base_url=os.getenv(
                "ARC_BASE_URL",
                "http://gateway:8001"
                if os.getenv("KAGGLE_IS_COMPETITION_RERUN")
                else "https://three.arcprize.org",
            ),
            environments_dir=os.getenv(
                "ARC_ENVIRONMENTS_DIR",
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
            ),
            recordings_dir=os.getenv(
                "ARC_RECORDINGS_DIR", "/kaggle/working/recordings"
            ),
            log_path=os.getenv(
                "ARC_VLM_LOG_PATH", "/kaggle/working/vlm_policy_run.jsonl"
            ),
            summary_path=os.getenv(
                "ARC_VLM_SUMMARY_PATH", "/kaggle/working/vlm_policy_summary.json"
            ),
            max_steps=int(os.getenv("ARC_VLM_MAX_STEPS", "30")),
            max_new_tokens=int(os.getenv("ARC_VLM_MAX_NEW_TOKENS", "850")),
            image_scale=int(os.getenv("ARC_VLM_IMAGE_SCALE", "8")),
            draw_grid=os.getenv("ARC_VLM_DRAW_GRID", "1") == "1",
            print_vlm_output=os.getenv("ARC_VLM_PRINT_OUTPUT", "1") == "1",
            allow_download=os.getenv("ALLOW_MODEL_DOWNLOAD", "0") == "1",
            force_reload_model=False,
            save_recording=os.getenv("ARC_SAVE_RECORDING", "1") == "1",
            actions_per_vlm_call=int(os.getenv("ARC_VLM_ACTIONS_PER_CALL", "3")),
            stop_sequence_on_no_change=True,
            stop_sequence_on_level_change=True,
            stop_sequence_on_actions_change=True,
            action6_max_candidates=int(os.getenv("ARC_VLM_ACTION6_MAX_CANDIDATES", "12")),
            action6_grid_points_per_axis=int(
                os.getenv("ARC_VLM_ACTION6_GRID_POINTS_PER_AXIS", "3")
            ),
            max_actions_per_vlm_call=int(
                os.getenv("ARC_VLM_MAX_ACTIONS_PER_CALL", "10")
            ),
            adaptive_action_planning=os.getenv("ARC_VLM_ADAPTIVE_ACTIONS", "1") == "1",
            repo_root=repo_root,
            in_kaggle=in_kaggle,
        )

    def with_overrides(self, **kwargs: object) -> "AppConfig":
        return replace(self, **kwargs)

    def ensure_output_dirs(self) -> None:
        Path(self.recordings_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.summary_path).parent.mkdir(parents=True, exist_ok=True)
