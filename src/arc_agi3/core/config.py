from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ClickExpansionConfig:
    grid_points_per_axis: int = 2
    max_candidates: int = 6


@dataclass(slots=True)
class BudgetConfig:
    max_steps_per_level: int = 128
    explore_phase_steps: int = 24
    revisit_limit: int = 3
    recent_state_window: int = 12
    novelty_patience_steps: int = 6
    semantic_patience_steps: int = 4
    recent_action_window: int = 6


@dataclass(slots=True)
class AgentConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    click_expansion: ClickExpansionConfig = field(default_factory=ClickExpansionConfig)
    enable_debug: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    backend: str = "mock"
    game_id: str = "ms00"
    mode: str = "offline"
    render_mode: str | None = None
    environments_dir: Path = Path("environment_files")
    recordings_dir: Path = Path("recordings")
    save_recording: bool = False
    include_frame_data: bool = True
