#!/usr/bin/env python3
"""
Minimal VLM-only ARC-AGI-3 runner with ARC API scorecard support.

Version: scorecard-debug-notebookload-2026-05-22

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
  export ARC_API_KEY=...
  python vlm_policy_runner.py --game-id ls20 --mode online --max-steps 30 --save-recording
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
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
    """Match the notebook: make transformers treat torchvision as unavailable.

    Some Kaggle runtimes have partially incompatible torchvision/Pillow builds.
    We do not need torchvision for this local VLM runner.
    """
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("USE_TORCHVISION", "0")
    os.environ.setdefault("DISABLE_TORCHVISION", "1")
    for module_name in list(sys.modules):
        if module_name.startswith("torchvision"):
            sys.modules.pop(module_name, None)
    try:
        import transformers.utils.import_utils as transformers_import_utils

        transformers_import_utils._torchvision_available = False
        transformers_import_utils._torchvision_version = "disabled"
    except Exception as exc:
        print("Could not patch transformers.utils.import_utils:", repr(exc))
    try:
        import transformers.utils as transformers_utils

        transformers_utils.is_torchvision_available = lambda: False
    except Exception as exc:
        print("Could not patch transformers.utils:", repr(exc))


def list_possible_model_dirs(root: str = "/kaggle/input", max_depth: int = 3) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    hits: list[Path] = []
    for config in root_path.rglob("config.json"):
        rel_depth = len(config.relative_to(root_path).parts)
        if rel_depth <= max_depth + 1:
            hits.append(config.parent)
    return sorted(hits)


def resolve_model_path(model_path: str | None = None) -> Path:
    """Resolve LOCAL_VLM_MODEL_PATH the same way as the capability notebook.

    It accepts either the exact HF model directory containing config.json or a
    parent directory if exactly one model config is found under it.
    """
    raw = str(model_path or os.getenv("LOCAL_VLM_MODEL_PATH", "")).strip()
    if not raw:
        candidates = list_possible_model_dirs()[:20]
        message = "Set LOCAL_VLM_MODEL_PATH or --model-path to the directory that contains config.json."
        if candidates:
            message += "\nCandidate model directories:\n" + "\n".join(f"- {p}" for p in candidates)
        raise ValueError(message)

    path = Path(raw).expanduser()
    allow_download = os.getenv("LOCAL_VLM_ALLOW_DOWNLOAD", "0") == "1"
    if allow_download and not path.exists():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if (path / "config.json").exists():
        return path

    candidates = sorted(p.parent for p in path.rglob("config.json"))
    if len(candidates) == 1:
        print("LOCAL_VLM_MODEL_PATH pointed to a parent directory; using detected model dir:", candidates[0])
        return candidates[0]
    if candidates:
        raise ValueError(
            "LOCAL_VLM_MODEL_PATH must point to one exact model folder containing config.json. "
            f"Got parent with {len(candidates)} candidates:\n" + "\n".join(f"- {p}" for p in candidates[:30])
        )

    visible = sorted(p.name for p in path.iterdir())[:40] if path.is_dir() else []
    raise ValueError(
        f"No config.json found under {path}. This is probably not a Hugging Face model folder. "
        f"Visible files/dirs: {visible}"
    )


class SplitVisionTextProcessor:
    """Notebook fallback processor: AutoTokenizer + AutoImageProcessor wrapper."""

    def __init__(self, tokenizer: Any, image_processor: Any):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        parts: list[str] = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("")
        if add_generation_prompt:
            parts.append("Answer:")
        return "\n".join(parts)

    def __call__(self, text=None, images=None, return_tensors: str = "pt"):
        if isinstance(text, list) and len(text) == 1:
            text_arg = text[0]
        else:
            text_arg = text
        encoded = self.tokenizer(text_arg, return_tensors=return_tensors)
        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
            encoded.update(image_inputs)
        return encoded

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def load_processor_with_fallback(model_path: Path, local_only: bool):
    """Use the exact fallback strategy from arc3-local-vlm-capability-check.ipynb."""
    from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=local_only,
            use_fast=False,
        )
    except Exception as processor_exc:
        print("AutoProcessor failed:", repr(processor_exc))
        print("Trying AutoTokenizer + AutoImageProcessor fallback...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=local_only,
    )
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=local_only,
            use_fast=False,
        )
    except Exception as image_exc:
        print("AutoImageProcessor failed:", repr(image_exc))
        image_processor = None
    return SplitVisionTextProcessor(tokenizer, image_processor)


def _import_qwen3_vl_model_class():
    """Return Qwen3VLForConditionalGeneration when this transformers build exposes it."""
    candidates = [
        ("transformers", "Qwen3VLForConditionalGeneration"),
        ("transformers.models.qwen3_vl", "Qwen3VLForConditionalGeneration"),
        ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLForConditionalGeneration"),
    ]
    import importlib

    last_exc = None
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"found {class_name} in {module_name}")
            return cls
        except Exception as exc:
            last_exc = exc
    print("Qwen3VLForConditionalGeneration import failed:", repr(last_exc))
    return None


def _from_pretrained_with_dtype_fallback(model_cls, model_path: str, common_kwargs: dict[str, Any]):
    """Load while handling transformers versions that prefer dtype vs torch_dtype."""
    try:
        return model_cls.from_pretrained(model_path, **common_kwargs)
    except TypeError as exc:
        if "dtype" in common_kwargs:
            retry_kwargs = dict(common_kwargs)
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            print("retrying model load with torch_dtype because dtype failed:", repr(exc))
            return model_cls.from_pretrained(model_path, **retry_kwargs)
        raise


def load_local_vlm(model_path: str | None = None):
    """Load local VLM robustly for Kaggle/Qwen3-VL.

    The capability notebook's processor fallback is kept, but model loading needs
    one extra case: Qwen3-VL is not a causal LM in transformers, so
    AutoModelForCausalLM cannot load Qwen3VLConfig. We first try the dedicated
    Qwen3VLForConditionalGeneration class, then AutoModelForVision2Seq, then
    AutoModelForCausalLM for non-Qwen models.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    disable_torchvision_for_transformers()

    model_path_obj = resolve_model_path(model_path)
    print("resolved model_path =", model_path_obj)

    local_only = os.getenv("LOCAL_VLM_ALLOW_DOWNLOAD", "0") != "1"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = load_processor_with_fallback(model_path_obj, local_only)

    common_kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        common_kwargs["device_map"] = "auto"

    config = AutoConfig.from_pretrained(
        str(model_path_obj),
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model_type = getattr(config, "model_type", None)
    arch_names = list(getattr(config, "architectures", []) or [])
    print("model_type:", model_type, "architectures:", arch_names)

    model = None

    # Qwen3-VL has a dedicated conditional-generation class. In some Kaggle
    # images AutoModelForVision2Seq is missing and AutoModelForCausalLM does not
    # recognize Qwen3VLConfig, so this path must come first.
    if model_type == "qwen3_vl" or "Qwen3VLForConditionalGeneration" in arch_names:
        qwen_cls = _import_qwen3_vl_model_class()
        if qwen_cls is not None:
            try:
                model = _from_pretrained_with_dtype_fallback(
                    qwen_cls,
                    str(model_path_obj),
                    common_kwargs,
                )
                print("loaded with Qwen3VLForConditionalGeneration")
            except Exception as qwen_exc:
                print("Qwen3VLForConditionalGeneration load failed:", repr(qwen_exc))

    if model is None:
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            try:
                model = _from_pretrained_with_dtype_fallback(
                    AutoModelForVision2Seq,
                    str(model_path_obj),
                    common_kwargs,
                )
                print("loaded with AutoModelForVision2Seq")
            except Exception as vision_exc:
                print("AutoModelForVision2Seq load failed:", repr(vision_exc))
        except Exception as import_exc:
            print("AutoModelForVision2Seq unavailable in this transformers build:", repr(import_exc))

    if model is None:
        if model_type == "qwen3_vl" or "Qwen3VLForConditionalGeneration" in arch_names:
            raise RuntimeError(
                "This transformers build has Qwen3-VL config but cannot import/load "
                "Qwen3VLForConditionalGeneration. Install a newer/source transformers "
                "build, or use the exact Kaggle image where the capability notebook loads "
                "this model. AutoModelForCausalLM cannot load Qwen3VLConfig."
            )
        model = _from_pretrained_with_dtype_fallback(
            AutoModelForCausalLM,
            str(model_path_obj),
            common_kwargs,
        )
        print("loaded with AutoModelForCausalLM")

    model.eval()
    print("loaded processor:", type(processor).__name__)
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


def resolve_arc_base_url() -> str:
    if os.getenv("ARC_BASE_URL"):
        return os.environ["ARC_BASE_URL"]
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        return "http://gateway:8001"
    return "https://three.arcprize.org"


def resolve_arc_api_key(mode: str) -> str:
    key = os.getenv("ARC_API_KEY")
    if key:
        return key
    if mode.lower() == "offline":
        # The local/offline wrapper path does not produce official online scorecards.
        # A dummy value keeps the arc-agi constructor happy for local experiments.
        return "test-key-123"
    raise RuntimeError(
        "ARC_API_KEY is required for --mode online/competition so the run can be "
        "submitted to the ARC API and connected to a scorecard. Set it with: "
        "export ARC_API_KEY=YOUR_KEY"
    )


def scorecard_url(base_url: str, scorecard_id: str) -> str:
    return f"{base_url.rstrip('/')}/scorecards/{scorecard_id}"


def parse_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return ["vlm-policy-runner", "vlm-only"]
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def print_scorecard_summary(card: Any, scorecard_id: str, base_url: str, mode: str) -> dict[str, Any]:
    summary = {
        "scorecard_id": getattr(card, "card_id", None) or scorecard_id,
        "score": getattr(card, "score", None),
        "total_environments_completed": getattr(card, "total_environments_completed", None),
        "total_environments": getattr(card, "total_environments", None),
        "total_levels_completed": getattr(card, "total_levels_completed", None),
        "total_actions": getattr(card, "total_actions", None),
        "scorecard_url": scorecard_url(base_url, scorecard_id),
        "replays_available": mode.lower() in {"online", "competition"},
    }

    print(f"scorecard_id={summary['scorecard_id']}")
    print(f"score={summary['score']}")
    if summary["total_environments_completed"] is not None:
        print(
            "environments_completed="
            f"{summary['total_environments_completed']}/{summary['total_environments']}"
        )
    print(f"levels_completed={summary['total_levels_completed']}")
    print(f"actions={summary['total_actions']}")
    print(f"scorecard_url={summary['scorecard_url']}")
    print("replays_available=" + ("yes" if summary["replays_available"] else "no"))
    return summary


def collect_game_reset(args: argparse.Namespace):
    from arc_agi import Arcade, OperationMode

    base_url = resolve_arc_base_url()
    api_key = resolve_arc_api_key(args.mode)
    recordings_dir = Path(args.recordings_dir)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    arcade = Arcade(
        arc_api_key=api_key,
        arc_base_url=base_url,
        operation_mode=OperationMode(args.mode.lower()),
        environments_dir=str(Path(args.environments_dir)),
        recordings_dir=str(recordings_dir),
    )

    scorecard_id = arcade.create_scorecard(
        source_url=args.source_url,
        tags=parse_tags(args.tags),
    )
    wrapper = arcade.make(
        args.game_id,
        scorecard_id=scorecard_id,
        save_recording=args.save_recording,
        include_frame_data=True,
        render_mode=args.render_mode,
    )
    if wrapper is None:
        raise RuntimeError(f"Failed to create ARC environment for {args.game_id}")

    raw = wrapper.reset()
    if raw is None:
        raise RuntimeError("ARC environment reset returned no frame")

    print("arc_mode:", args.mode)
    print("arc_base_url:", base_url)
    print("scorecard_id:", scorecard_id)
    print("scorecard_url:", scorecard_url(base_url, scorecard_id))
    print("replays_available:", "yes" if args.mode.lower() in {"online", "competition"} else "no")
    return arcade, wrapper, scorecard_id, base_url, raw

def ask_vlm_for_action(
    processor: Any,
    model: Any,
    raw_frame: Any,
    history: list[dict[str, Any]],
    game_id: str,
    max_new_tokens: int,
    print_vlm_output: bool = False,
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
    if print_vlm_output:
        print("\n" + "-" * 30 + " RAW VLM OUTPUT " + "-" * 30)
        print(raw_text)
        print("-" * 78 + "\n")
    parsed = extract_json_object(raw_text)
    action_text = str(parsed["chosen_action"]).strip()
    validate_action_text(action_text, raw_frame)
    return action_text, parsed, raw_text


def run_episode(args: argparse.Namespace) -> list[dict[str, Any]]:
    processor, model = load_local_vlm(args.model_path)
    arcade, wrapper, scorecard_id, base_url, raw = collect_game_reset(args)

    history: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    try:
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
                    print_vlm_output=args.print_vlm_output,
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
                    print_vlm_output=args.print_vlm_output,
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

        final_summary = None
        if not args.keep_scorecard_open:
            try:
                card = arcade.close_scorecard(scorecard_id)
                final_summary = print_scorecard_summary(
                    card=card,
                    scorecard_id=scorecard_id,
                    base_url=base_url,
                    mode=args.mode,
                )
            except Exception as exc:
                print("close_scorecard failed:", repr(exc))
        else:
            print("scorecard left open:", scorecard_id)
            print("scorecard_url:", scorecard_url(base_url, scorecard_id))

        if args.summary_path:
            summary_path = Path(args.summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "game_id": args.game_id,
                "mode": args.mode,
                "scorecard_id": scorecard_id,
                "scorecard_url": scorecard_url(base_url, scorecard_id),
                "steps_logged": len(logs),
                "final_scorecard": final_summary,
            }
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("wrote summary:", summary_path)

    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal VLM-only ARC-AGI-3 runner")
    parser.add_argument("--game-id", default=os.getenv("ARC_VLM_GAME_ID", "ls20"))
    parser.add_argument("--mode", default=os.getenv("ARC_VLM_MODE", "online"), choices=["offline", "online", "competition"])
    parser.add_argument("--render-mode", default=os.getenv("ARC_VLM_RENDER_MODE") or None)
    parser.add_argument("--environments-dir", default=os.getenv("ARC_ENVIRONMENTS_DIR", str(ENV_DIR)))
    parser.add_argument(
        "--recordings-dir",
        default=os.getenv("ARC_RECORDINGS_DIR", str(Path("/kaggle/working/recordings") if IN_KAGGLE else Path("recordings"))),
    )
    parser.add_argument("--model-path", default=os.getenv("LOCAL_VLM_MODEL_PATH") or None)
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("ARC_VLM_MAX_STEPS", "30")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LOCAL_VLM_MAX_NEW_TOKENS", "384")))
    parser.add_argument("--log-path", default=os.getenv("ARC_VLM_LOG_PATH", "vlm_policy_run.jsonl"))
    parser.add_argument("--summary-path", default=os.getenv("ARC_VLM_SUMMARY_PATH", "vlm_policy_summary.json"))
    parser.add_argument(
        "--source-url",
        default=os.getenv("ARC_SCORECARD_SOURCE_URL", "https://github.com/use08174/agi-arc"),
    )
    parser.add_argument("--tags", default=os.getenv("ARC_SCORECARD_TAGS", "vlm-policy-runner,vlm-only"))
    parser.add_argument("--save-recording", action="store_true")
    parser.add_argument(
        "--print-vlm-output",
        action="store_true",
        default=os.getenv("ARC_VLM_PRINT_OUTPUT", "0") == "1",
        help="Print raw VLM text before JSON parsing. Also useful when parsing fails.",
    )
    parser.add_argument("--keep-scorecard-open", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_episode(parse_args())
