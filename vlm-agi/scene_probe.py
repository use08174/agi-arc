from __future__ import annotations
from typing import Any

from grid import frame_metadata, grid_to_rgb_array, is_valid_grid, latest_grid_from_raw
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

    print(f"[scene {label}]")
    if not is_valid_grid(grid):
        print("Skipping scene probe because the current frame grid is empty or invalid.")
        return None

    current_image = grid_to_rgb_array(
        grid,
        scale=runner.config.image_scale,
        draw_grid=False,
    )
    if display_image:
        maybe_display_image(current_image)

    raw_scene_text = runner.vlm.generate_scene_understanding(
        image=current_image,
        prompt=build_compact_scene_prompt(raw_frame, runner.config),
        max_new_tokens=500,
    )
    try:
        parsed = extract_json_object(raw_scene_text)
        summary = parsed.get("game_objective_hypothesis") or parsed.get("player", {}).get("why")
        if summary:
            print(summary)
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
        record = runner.step_once(display_after=display_after)
        analysis = run_scene_understanding_probe(
            runner, label=f"after step_once #{index + 1}"
        )
        results.append({"step": index + 1, "record": record, "analysis": analysis})
        meta = frame_metadata(runner.session["raw"], game_id=runner.config.game_id)
        if analysis is None:
            print("Stopping because the current frame is invalid after the last action.")
            break
        if meta.get("state") != "NOT_FINISHED":
            print(f"Stopping because state changed: {meta.get('state')}")
            break

    print("scene probe done")
    return results
