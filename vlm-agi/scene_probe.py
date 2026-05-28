from __future__ import annotations

import json
from typing import Any

from grid import frame_metadata, grid_to_rgb_array, latest_grid_from_raw
from model import extract_json_object
from prompts import build_compact_scene_prompt
from session import VLMArcRunner


def maybe_display_image(image: Any) -> None:
    try:
        from IPython.display import display
        from PIL import Image

        display(Image.fromarray(image))
    except Exception:
        pass


def run_scene_understanding_probe(
    runner: VLMArcRunner,
    *,
    label: str = "current",
    display_image: bool = True,
) -> dict[str, Any] | None:
    raw_frame = runner.session["raw"]
    grid = latest_grid_from_raw(raw_frame)
    meta = frame_metadata(raw_frame, game_id=runner.config.game_id)
    current_image = grid_to_rgb_array(
        grid,
        scale=runner.config.image_scale,
        draw_grid=False,
    )

    print("=" * 100)
    print(f"SCENE PROBE: {label}")
    print("metadata:")
    print(
        json.dumps(
            {
                "state": meta.get("state"),
                "levels_completed": meta.get("levels_completed"),
                "available_actions": meta.get("available_actions"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if display_image:
        maybe_display_image(current_image)

    raw_scene_text = runner.vlm.generate_scene_understanding(
        image=current_image,
        prompt=build_compact_scene_prompt(raw_frame, runner.config),
        max_new_tokens=500,
    )
    print("RAW VLM OUTPUT:")
    print(raw_scene_text)

    try:
        parsed = extract_json_object(raw_scene_text)
        print("\nPARSED JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        return parsed
    except Exception as exc:
        print("\nFailed to parse JSON:", repr(exc))
        return None


def run_scene_probe_sequence(
    runner: VLMArcRunner,
    *,
    steps: int = 5,
    display_after: bool = True,
) -> list[dict[str, Any]]:
    results = [{"step": "initial", "analysis": run_scene_understanding_probe(runner, label="initial")}]

    for index in range(steps):
        print("\n" + "#" * 100)
        print(f"RUNNING step_once #{index + 1}")
        print("#" * 100)
        record = runner.step_once(display_after=display_after)
        print("ACTION RECORD:")
        print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        analysis = run_scene_understanding_probe(
            runner, label=f"after step_once #{index + 1}"
        )
        results.append({"step": index + 1, "record": record, "analysis": analysis})
        meta = frame_metadata(runner.session["raw"], game_id=runner.config.game_id)
        if meta.get("state") != "NOT_FINISHED":
            print(f"Stopping because state changed: {meta.get('state')}")
            break

    print("\nDONE. scene_probe_results contains all analyses.")
    return results
