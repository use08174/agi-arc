from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _runner():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "vlm-agi"))
    session_module = importlib.import_module("session")
    config_module = importlib.import_module("config")

    config = config_module.AppConfig(
        model_path="model",
        game_id="bp35",
        arc_mode="online",
        arc_api_key="",
        arc_base_url="https://three.arcprize.org",
        environments_dir="environment_files",
        recordings_dir="recordings",
        log_path="vlm.jsonl",
        summary_path="summary.json",
        max_steps=30,
        max_new_tokens=256,
        image_scale=8,
        draw_grid=True,
        print_vlm_output=False,
        allow_download=False,
        force_reload_model=False,
        save_recording=False,
        actions_per_vlm_call=3,
        stop_sequence_on_no_change=True,
        stop_sequence_on_level_change=True,
        stop_sequence_on_actions_change=True,
        action6_max_candidates=8,
        action6_grid_points_per_axis=3,
        max_actions_per_vlm_call=10,
        adaptive_action_planning=True,
        repo_root=Path("."),
        in_kaggle=False,
    )
    return session_module.VLMArcRunner(config, vlm=None)


def test_expand_action_sequence_repeats_stable_move():
    runner = _runner()
    runner.session["logs"] = [
        {"action": "ACTION4", "test_result": "visible_change_no_progress"},
        {"action": "ACTION4", "test_result": "visible_change_no_progress"},
    ]

    expanded = runner.maybe_expand_action_sequence(
        ["ACTION4"],
        {"progress_assessment": "progress"},
        5,
    )

    assert expanded == ["ACTION4", "ACTION4", "ACTION4", "ACTION4", "ACTION4"]


def test_expand_action_sequence_does_not_repeat_unstable_action():
    runner = _runner()
    runner.session["logs"] = [
        {"action": "ACTION4", "test_result": "no_visible_effect"},
        {"action": "ACTION4", "test_result": "no_visible_effect"},
    ]

    expanded = runner.maybe_expand_action_sequence(
        ["ACTION4"],
        {"progress_assessment": "no_progress"},
        5,
    )

    assert expanded == ["ACTION4"]
