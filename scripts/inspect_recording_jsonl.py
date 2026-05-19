#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an ARC recording JSONL file")
    parser.add_argument("path", type=Path, help="Path to *.recording.jsonl")
    parser.add_argument("--list-steps", action="store_true", help="Print one summary line per recorded step")
    parser.add_argument("--step", type=int, default=None, help="Render one step as ASCII")
    parser.add_argument("--show-last", action="store_true", help="Render the last step as ASCII")
    parser.add_argument("--max-grid-size", type=int, default=64, help="Maximum rows/cols to render")
    return parser.parse_args()


def load_recording_jsonl(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for step_idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = obj.get("data", {}) or {}
            frames.append(
                {
                    "step": step_idx,
                    "timestamp": obj.get("timestamp"),
                    "game_id": data.get("game_id"),
                    "state": data.get("state"),
                    "levels_completed": data.get("levels_completed", 0),
                    "win_levels": data.get("win_levels", 0),
                    "action": _extract_action(data),
                    "grid": normalize_frame_to_single_grid(data.get("frame")),
                }
            )
    return frames


def normalize_frame_to_single_grid(frame: Any) -> list[list[int]] | None:
    if frame is None:
        return None
    current = frame
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    if not isinstance(current, list) or not current:
        return None
    if not isinstance(current[0], list):
        return None
    if current and current[0] and isinstance(current[0][0], list):
        current = current[-1]
    try:
        return [[int(cell) for cell in row] for row in current]
    except Exception:
        return None


def summarize(frames: list[dict[str, Any]], path: Path) -> None:
    if not frames:
        print(f"path={path}")
        print("frames=0")
        return
    first = frames[0]
    last = frames[-1]
    actions = [frame["action"] for frame in frames if frame.get("action")]
    grid = next((frame["grid"] for frame in frames if frame.get("grid")), None)
    height = len(grid) if grid else 0
    width = len(grid[0]) if grid and grid[0] else 0
    print(f"path={path}")
    print(f"game_id={first.get('game_id')}")
    print(f"frames={len(frames)}")
    print(f"grid={width}x{height}")
    print(f"initial_state={first.get('state')}")
    print(f"final_state={last.get('state')}")
    print(f"levels_completed={last.get('levels_completed', 0)}/{last.get('win_levels', 0)}")
    print(f"actions_recorded={len(actions)}")
    if actions:
        print("action_sequence=" + ", ".join(actions))


def print_steps(frames: list[dict[str, Any]]) -> None:
    for frame in frames:
        print(
            f"step={frame['step']} state={frame.get('state')} "
            f"levels={frame.get('levels_completed', 0)}/{frame.get('win_levels', 0)} "
            f"action={frame.get('action') or '-'}"
        )


def render_ascii(grid: list[list[int]] | None, max_grid_size: int) -> None:
    if not grid:
        print("no grid")
        return
    palette = " .123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = grid[:max_grid_size]
    for row in rows:
        chars = []
        for cell in row[:max_grid_size]:
            value = int(cell)
            chars.append(palette[value] if 0 <= value < len(palette) else "?")
        print("".join(chars))


def _extract_action(data: dict[str, Any]) -> str | None:
    action = data.get("action")
    if isinstance(action, dict):
        name = action.get("name") or action.get("action")
        payload = action.get("data") or action.get("payload") or {}
        if isinstance(name, str) and isinstance(payload, dict) and payload:
            suffix = ",".join(f"{key}={payload[key]}" for key in sorted(payload))
            return f"{name}|{suffix}"
        if isinstance(name, str):
            return name
    if isinstance(action, str):
        return action
    return None


def main() -> None:
    args = parse_args()
    frames = load_recording_jsonl(args.path)
    summarize(frames, args.path)
    if args.list_steps:
        print_steps(frames)
    target_step = len(frames) - 1 if args.show_last and frames else args.step
    if target_step is not None:
        if target_step < 0 or target_step >= len(frames):
            raise SystemExit(f"step out of range: {target_step}")
        frame = frames[target_step]
        print(f"\nrender_step={frame['step']} state={frame.get('state')} action={frame.get('action') or '-'}")
        render_ascii(frame.get("grid"), args.max_grid_size)


if __name__ == "__main__":
    main()
