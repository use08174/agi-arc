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
class LLMConfig:
    enabled: bool = False
    mode: str = "rank_only"
    provider: str = "noop"
    model: str = ""
    model_path: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_ranked_actions: int = 3
    include_hypotheses: bool = False
    max_new_tokens: int = 256
    thinking_mode: str = "brief"
    thinking_max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    start_step: int = 1
    step_interval: int = 1
    max_calls_per_episode: int = 80
    cache_enabled: bool = True
    trace_enabled: bool = True
    trace_print_prompt: bool = False
    trace_print_raw_response: bool = True
    trace_print_limit: int = 6


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
