#!/usr/bin/env python3
"""
Minimal VLM-only ARC-AGI-3 runner.

Flow:
  current frame image + metadata + recent history
  -> local VLM returns JSON with one chosen_action
  -> validate/parse action
  -> wrapper.step(action)
  -> repeat

Environment variables commonly used:
  LOCAL_VLM_MODEL_PATH=/path/to/local/vlm
  LOCAL_VLM_ALLOW_DOWNLOAD=0
  ARC_API_KEY=...
  ARC_BASE_URL=...
  ARC_VLM_IMAGE_SCALE=14
  LOCAL_VLM_MAX_NEW_TOKENS=384

Example:
  python vlm_policy_runner.py --game-id ls20 --mode offline --max-steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Path / environment setup
# -----------------------------------------------------------------------------

IN_KAGGLE = Path("/kaggle/input").exists()
COMP_ROOT = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
ENV_DIR = next(
    (
        p
        for p in [
            COMP_ROOT / "environment_files",
            Path.cwd() / "environment_files",
            Path("/kaggle/working/agi-arc/environment_files"),
        ]
        if p.exists()
    ),
    COMP_ROOT / "environment_files",
)
REPO_ROOT = Path(os.getenv("AGI_ARC_REPO", Path.cwd()))

for candidate in [REPO_ROOT / "src", Path.cwd() / "src", Path("/kaggle/working/agi-arc/src")]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("USE_TORCHVISION", "0")
os.environ.setdefault("DISABLE_TORCHVISION", "1")


# -----------------------------------------------------------------------------
# Grid / image helpers
# -----------------------------------------------------------------------------

ARC_PALETTE = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 80, 220),
    6: (255, 133, 27),
    7: (127, 219, 255),
    8: (135, 135, 135),
    9: (255, 255, 255),
    10: (80, 80, 80),
    11: (180, 180, 180),
    12: (128, 0, 0),
    13: (0, 128, 128),
    14: (128, 128, 0),
    15: (255, 105, 180),
}


def as_list_grid(grid: Any) -> list[list[int]]:
    if grid is None:
        return []
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [[int(cell) for cell in row] for row in grid]


def raw_frame_stack(raw_frame: Any) -> list[Any]:
    return list(getattr(raw_frame, "frame", []) or [])


def latest_grid_from_raw(raw_frame: Any) -> list[list[int]]:
    frames = raw_frame_stack(raw_frame)
    return as_list_grid(frames[-1]) if frames else []


def summarize_grid(grid: Any) -> dict[str, Any]:
    grid = as_list_grid(grid)
    if not grid:
        return {"shape": [0, 0], "colors": [], "nonzero": 0}
    arr = np.asarray(grid, dtype=int)
    return {
        "shape": list(arr.shape),
        "colors": sorted(int(x) for x in np.unique(arr)),
        "nonzero": int(np.count_nonzero(arr)),
    }


def grid_to_rgb_array(grid: Any) -> np.ndarray:
    grid = as_list_grid(grid)
    if not grid:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    arr = np.asarray(grid, dtype=int)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for value, color in ARC_PALETTE.items():
        rgb[arr == value] = color
    return rgb


def grid_to_image(grid: Any, scale: int = 14, draw_grid: bool = True) -> Image.Image:
    rgb = grid_to_rgb_array(grid)
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb, mode="RGB").resize(
        (w * scale, h * scale),
        resample=Image.Resampling.NEAREST,
    )
    if draw_grid and scale >= 8:
        px = img.load()
        line = (35, 35, 35)
        for gx in range(0, w * scale, scale):
            for yy in range(h * scale):
                px[gx, yy] = line
        for gy in range(0, h * scale, scale):
            for xx in range(w * scale):
                px[xx, gy] = line
    return img


def frame_metadata(raw_frame: Any, game_id: str | None = None) -> dict[str, Any]:
    grid = latest_grid_from_raw(raw_frame)
    state = getattr(raw_frame, "state", None)
    state_name = getattr(state, "name", None) or str(state)
    return {
        "game_id": game_id or getattr(raw_frame, "game_id", None),
        "state": state_name,
        "available_actions": [
            getattr(a, "name", str(a))
            for a in list(getattr(raw_frame, "available_actions", []) or [])
        ],
        "levels_completed": getattr(raw_frame, "levels_completed", None),
        "win_levels": getattr(raw_frame, "win_levels", None),
        "frame_count": len(raw_frame_stack(raw_frame)),
        "latest_grid": summarize_grid(grid),
    }


# -----------------------------------------------------------------------------
# VLM loading / generation
# -----------------------------------------------------------------------------


def disable_torchvision_for_transformers() -> None:
    for module_name in list(sys.modules):
        if module_name.startswith("torchvision"):
            sys.modules.pop(module_name, None)
    try:
        import transformers.utils.import_utils as import_utils

        import_utils._torchvision_available = False
        import_utils._torchvision_version = "disabled"
    except Exception:
        pass
    try:
        import transformers.utils as transformers_utils

        transformers_utils.is_torchvision_available = lambda: False
    except Exception:
        pass


def load_local_vlm(model_path: str | None = None):
    import torch
    from transformers import AutoConfig, AutoProcessor

    disable_torchvision_for_transformers()

    raw_model_path = model_path or os.getenv("LOCAL_VLM_MODEL_PATH")
    if not raw_model_path:
        raise ValueError("Set LOCAL_VLM_MODEL_PATH or pass --model-path.")

    model_path_obj = Path(raw_model_path)
    allow_download = os.getenv("LOCAL_VLM_ALLOW_DOWNLOAD", "0") == "1"
    if not allow_download and not model_path_obj.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path_obj}")

    local_only = not allow_download

    processor = AutoProcessor.from_pretrained(
        str(model_path_obj),
        trust_remote_code=True,
        local_files_only=local_only,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        str(model_path_obj),
        trust_remote_code=True,
        local_files_only=local_only,
    )

    model_type = getattr(config, "model_type", None)
    print("model_type:", model_type, "architectures:", list(getattr(config, "architectures", []) or []))

    common_kwargs = dict(
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        common_kwargs["device_map"] = "auto"

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path_obj), **common_kwargs)
    elif model_type == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(str(model_path_obj), **common_kwargs)
    else:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(str(model_path_obj), **common_kwargs)

    model.eval()
    print("loaded model:", type(model).__name__)
    return processor, model


def model_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


POLICY_SYSTEM_PROMPT = """
You are controlling an ARC-AGI-3 interactive grid game.
Coordinates use x,y with (0,0) at top-left.
Actions are abstract and game-specific. ACTION6 may require x,y coordinates.

Your task: choose exactly ONE next action.

Return STRICT JSON only. Do not include markdown or text outside JSON.

Allowed action formats:
- ACTION1
- ACTION2
- ACTION3
- ACTION4
- ACTION5
- ACTION6
- ACTION6|x=<int>,y=<int>

Rules:
- Only choose an action whose base name appears in available_actions.
- If choosing ACTION6 with coordinates, keep x,y inside the current grid bounds.
- Prefer actions that make progress or reveal the game rule.
- Avoid repeating recent actions that caused no visible change.
- If uncertain, choose the most informative low-risk probe.
""".strip()


def build_policy_prompt(meta: dict[str, Any], history: list[dict[str, Any]], max_history: int = 8) -> str:
    recent = history[-max_history:]
    return f"""
Current metadata:
{json.dumps(meta, ensure_ascii=False, indent=2)}

Recent action history:
{json.dumps(recent, ensure_ascii=False, indent=2)}

Choose the next single action.

Return exactly this JSON shape:
{{
  "analysis": "brief visual facts and current situation",
  "goal_hypothesis": "your best guess for the goal",
  "chosen_action": "ACTION3",
  "expected_result": "what you expect to change",
  "reason": "why this action is useful now",
  "confidence": 0.0
}}
""".strip()


def prepare_vlm_inputs(processor: Any, image: Image.Image, prompt: str):
    content = [
        {"type": "image"},
        {"type": "text", "text": POLICY_SYSTEM_PROMPT + "\n\n" + prompt},
    ]
    messages = [{"role": "user", "content": content}]

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = POLICY_SYSTEM_PROMPT + "\n\n" + prompt

    try:
        return processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        return processor(images=[image], text=[text], return_tensors="pt")


def generate_local_vlm(processor: Any, model: Any, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = prepare_vlm_inputs(processor, image, prompt)
    device = model_device(model)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    new_tokens = generated[:, input_len:] if input_len else generated

    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# -----------------------------------------------------------------------------
# Action parsing / validation / transition summary
# -----------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in VLM output:\n{text}")
    return json.loads(match.group(0))


def available_action_names(raw_frame: Any) -> list[str]:
    return [
        getattr(a, "name", str(a))
        for a in list(getattr(raw_frame, "available_actions", []) or [])
    ]


def validate_action_text(action_text: str, raw_frame: Any) -> None:
    names = available_action_names(raw_frame)
    if "|" in action_text:
        name, payload_text = action_text.split("|", 1)
    else:
        name, payload_text = action_text, ""

    name = name.strip()
    if name not in names:
        raise ValueError(f"Action {name!r} is not in available_actions={names}")

    if payload_text:
        parts = {}
        for item in payload_text.split(","):
            key, value = item.split("=", 1)
            parts[key.strip()] = int(value)

        if "x" in parts or "y" in parts:
            grid = latest_grid_from_raw(raw_frame)
            h = len(grid)
            w = len(grid[0]) if h else 0
            x = parts.get("x")
            y = parts.get("y")
            if x is None or y is None:
                raise ValueError(f"Coordinate payload requires both x and y: {action_text}")
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"Coordinate out of bounds: x={x}, y={y}, grid={w}x{h}")


def parse_action(action_text: str):
    from arcengine import GameAction

    action_text = action_text.strip()
    if "|" not in action_text:
        return GameAction.from_name(action_text), None

    name, payload_text = action_text.split("|", 1)
    payload = {}
    for part in payload_text.split(","):
        key, value = part.split("=", 1)
        payload[key.strip()] = int(value)
    return GameAction.from_name(name.strip()), payload


def summarize_transition(before_grid: Any, after_grid: Any, after_raw: Any) -> dict[str, Any]:
    before = np.asarray(as_list_grid(before_grid), dtype=int)
    after = np.asarray(as_list_grid(after_grid), dtype=int)

    summary: dict[str, Any] = {
        "before_shape": list(before.shape) if before.size else [0, 0],
        "after_shape": list(after.shape) if after.size else [0, 0],
        "state": getattr(getattr(after_raw, "state", None), "name", str(getattr(after_raw, "state", None))),
        "levels_completed": getattr(after_raw, "levels_completed", None),
        "win_levels": getattr(after_raw, "win_levels", None),
    }

    if before.shape == after.shape and before.size:
        diff = before != after
        ys, xs = np.where(diff)
        summary["changed_cells"] = int(diff.sum())
        summary["changed_bbox"] = (
            {
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
            }
            if len(xs)
            else None
        )
    else:
        summary["changed_cells"] = "shape_changed_or_empty"
        summary["changed_bbox"] = None

    return summary


def is_terminal(raw_frame: Any) -> bool:
    state = getattr(raw_frame, "state", None)
    state_name = (getattr(state, "name", None) or str(state) or "").upper()
    if state_name in {"WIN", "DONE", "GAME_OVER", "LOSE", "LOST"}:
        return True
    levels_completed = getattr(raw_frame, "levels_completed", None)
    win_levels = getattr(raw_frame, "win_levels", None)
    return bool(win_levels is not None and levels_completed is not None and levels_completed >= win_levels)


# -----------------------------------------------------------------------------
# ARC setup / VLM-only episode loop
# -----------------------------------------------------------------------------


def collect_game_reset(game_id: str, mode: str, render_mode: str | None, save_recording: bool):
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        arc_api_key=os.getenv("ARC_API_KEY", "test-key-123"),
        arc_base_url=os.getenv(
            "ARC_BASE_URL",
            "http://gateway:8001" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") else "https://three.arcprize.org",
        ),
        operation_mode=OperationMode(mode.lower()),
        environments_dir=str(ENV_DIR),
        recordings_dir=str(Path("/kaggle/working/recordings") if IN_KAGGLE else Path("recordings")),
    )
    scorecard_id = arcade.create_scorecard(source_url="vlm-policy-runner", tags=["vlm-policy-runner"])
    wrapper = arcade.make(
        game_id,
        scorecard_id=scorecard_id,
        save_recording=save_recording,
        include_frame_data=True,
        render_mode=render_mode,
    )
    raw = wrapper.reset()
    return arcade, wrapper, scorecard_id, raw


def ask_vlm_for_action(
    processor: Any,
    model: Any,
    raw_frame: Any,
    history: list[dict[str, Any]],
    game_id: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any], str]:
    grid = latest_grid_from_raw(raw_frame)
    image = grid_to_image(
        grid,
        scale=int(os.getenv("ARC_VLM_IMAGE_SCALE", "14")),
        draw_grid=True,
    )
    meta = frame_metadata(raw_frame, game_id=game_id)
    prompt = build_policy_prompt(meta, history)
    raw_text = generate_local_vlm(processor, model, image, prompt, max_new_tokens=max_new_tokens)
    parsed = extract_json_object(raw_text)
    action_text = str(parsed["chosen_action"]).strip()
    validate_action_text(action_text, raw_frame)
    return action_text, parsed, raw_text


def run_episode(args: argparse.Namespace) -> list[dict[str, Any]]:
    processor, model = load_local_vlm(args.model_path)
    arcade, wrapper, scorecard_id, raw = collect_game_reset(
        game_id=args.game_id,
        mode=args.mode,
        render_mode=args.render_mode,
        save_recording=args.save_recording,
    )

    history: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    try:
        print("scorecard_id:", scorecard_id)
        print("initial_meta:", json.dumps(frame_metadata(raw, args.game_id), ensure_ascii=False, indent=2))

        for step_idx in range(args.max_steps):
            before_grid = latest_grid_from_raw(raw)

            try:
                action_text, plan, raw_vlm_output = ask_vlm_for_action(
                    processor=processor,
                    model=model,
                    raw_frame=raw,
                    history=history,
                    game_id=args.game_id,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                # One retry with the error included in history. This is still VLM-driven;
                # it just asks the model to fix formatting/action validity.
                retry_history = history + [
                    {
                        "step": step_idx,
                        "error": repr(exc),
                        "instruction": "Previous output was invalid. Return strict JSON and choose one available action only.",
                    }
                ]
                action_text, plan, raw_vlm_output = ask_vlm_for_action(
                    processor=processor,
                    model=model,
                    raw_frame=raw,
                    history=retry_history,
                    game_id=args.game_id,
                    max_new_tokens=args.max_new_tokens,
                )

            action, payload = parse_action(action_text)
            after_raw = wrapper.step(action, data=payload)
            after_grid = latest_grid_from_raw(after_raw)
            transition = summarize_transition(before_grid, after_grid, after_raw)

            short_record = {
                "step": step_idx,
                "action": action_text,
                "reason": plan.get("reason"),
                "expected_result": plan.get("expected_result"),
                "confidence": plan.get("confidence"),
                "transition": transition,
            }
            history.append(short_record)

            full_record = {
                **short_record,
                "plan": plan,
                "raw_vlm_output": raw_vlm_output,
                "after_meta": frame_metadata(after_raw, args.game_id),
            }
            logs.append(full_record)

            print("=" * 80)
            print(f"step={step_idx} action={action_text} confidence={plan.get('confidence')}")
            print("reason:", plan.get("reason"))
            print("transition:", json.dumps(transition, ensure_ascii=False))

            raw = after_raw
            if is_terminal(raw):
                print("episode ended:", json.dumps(frame_metadata(raw, args.game_id), ensure_ascii=False, indent=2))
                break

    finally:
        if args.log_path:
            log_path = Path(args.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                for row in logs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("wrote logs:", log_path)

        if not args.keep_scorecard_open:
            try:
                arcade.close_scorecard(scorecard_id)
                print("closed scorecard:", scorecard_id)
            except Exception as exc:
                print("close_scorecard failed:", repr(exc))

    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal VLM-only ARC-AGI-3 runner")
    parser.add_argument("--game-id", default=os.getenv("ARC_VLM_GAME_ID", "ls20"))
    parser.add_argument("--mode", default=os.getenv("ARC_VLM_MODE", "offline"), choices=["offline", "online", "competition"])
    parser.add_argument("--render-mode", default=os.getenv("ARC_VLM_RENDER_MODE") or None)
    parser.add_argument("--model-path", default=os.getenv("LOCAL_VLM_MODEL_PATH") or None)
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("ARC_VLM_MAX_STEPS", "30")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LOCAL_VLM_MAX_NEW_TOKENS", "384")))
    parser.add_argument("--log-path", default=os.getenv("ARC_VLM_LOG_PATH", "vlm_policy_run.jsonl"))
    parser.add_argument("--save-recording", action="store_true")
    parser.add_argument("--keep-scorecard-open", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_episode(parse_args())


