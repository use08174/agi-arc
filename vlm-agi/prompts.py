from __future__ import annotations

import json
from typing import Any

from config import AppConfig
from grid import frame_metadata, latest_grid_from_raw, summarize_objects


SCENE_UNDERSTANDING_SYSTEM_PROMPT = """
You are analyzing an abstract 2D grid game screen.

Your job is to briefly identify the visible game elements and infer the likely game goal.
Do not choose an action.

Rules:
- Only mention objects that are clearly visible.
- Do not invent missing objects.
- Do not infer symmetric UI elements.
- Do not list tiny noise unless it seems important.
- For each element, say what it likely is in one short sentence.
- Separate observation from uncertainty.
- If unsure, say "unknown" rather than inventing a role.
- Do not describe objects as animals, faces, bodies, whales, fish, or creatures.

Return strict JSON only.
""".strip()


POLICY_SYSTEM_PROMPT = """
You are playing an abstract 2D grid game made of colored geometric shapes.

You may receive one image or two images:
- If one image is provided, infer the current game state from it.
- If two images are provided, Image 1 is BEFORE the previous action and Image 2 is AFTER/current.

When two images are provided, your first job is to compare them.
Do not restart from scratch.
First identify what changed between Image 1 and Image 2:
- Which object moved?
- In what direction did it move?
- Which cells or shapes appeared, disappeared, or shifted?
- Did the previous action make meaningful progress or little/no progress?

The object or region that changed after the previous action is the strongest evidence for the controlled player.
Use visual change to update your player hypothesis.
Do not choose the next action until you have reasoned about the visual difference.

Controls:
ACTION1=up
ACTION2=down
ACTION3=left
ACTION4=right
ACTION5=interact/special
ACTION6=x,y coordinate
ACTION7=extra

Do not describe the image as a character, animal, face, body, whale, fish, or creature.

Return strict JSON only:
{
  "visual_change": "what changed from Image 1 to Image 2, or 'only one image' if no previous image",
  "player_hypothesis": "which object is likely controlled and why",
  "progress_assessment": "progress / no_progress / uncertain",
  "reasoning": "brief reason for the next action",
  "chosen_actions": ["ACTION1"]
}
""".strip()


def compact_transition_for_prompt(transition: Any) -> dict[str, Any] | None:
    if not isinstance(transition, dict):
        return None
    return {
        "changed_cells": transition.get("changed_cells"),
        "changed_bbox": transition.get("changed_bbox"),
        "levels_completed": transition.get("levels_completed"),
        "state": transition.get("state"),
    }


def build_compact_scene_prompt(raw_frame: Any, config: AppConfig) -> str:
    grid = latest_grid_from_raw(raw_frame)
    meta = frame_metadata(raw_frame, game_id=config.game_id)
    current_meta = {
        "game_id": meta.get("game_id"),
        "state": meta.get("state"),
        "levels_completed": meta.get("levels_completed"),
        "available_actions": meta.get("available_actions"),
    }
    detected = summarize_objects(grid, max_objects=16)
    return f"""
Analyze this current game screen.

Current metadata:
{json.dumps(current_meta, ensure_ascii=False)}

Detected connected color components:
{json.dumps(detected, ensure_ascii=False)}

Return JSON with this schema:
{{
  "elements": [
    {{
      "element": "short visible element name",
      "location": "approximate location",
      "role": "what this element likely is, in one sentence",
      "confidence": "low/medium/high"
    }}
  ],
  "player": {{
    "element": "most likely player or unknown",
    "why": "short reason",
    "confidence": "low/medium/high"
  }},
  "goal": {{
    "element": "most likely goal/target or unknown",
    "why": "short reason",
    "confidence": "low/medium/high"
  }},
  "game_objective_hypothesis": "one or two sentences describing what the player probably needs to do"
}}
""".strip()


def build_policy_prompt(
    meta: dict[str, Any],
    session: dict[str, Any],
    config: AppConfig,
    *,
    action6_candidates: list[dict[str, Any]] | None = None,
) -> str:
    current_available = list(meta.get("available_actions", []))
    prev_reasoning = session.get("last_vlm_reasoning")
    prev_visual_change = session.get("last_visual_change")
    prev_player_hypothesis = session.get("last_player_hypothesis")
    prev_action = session.get("last_action")
    prev_transition = compact_transition_for_prompt(session.get("last_transition"))

    if prev_action or prev_transition:
        previous_context = f"""
Previous action:
{prev_action or "(none)"}

Previous VLM visual_change:
{prev_visual_change or "(none)"}

Previous VLM player_hypothesis:
{prev_player_hypothesis or "(none)"}

Previous reasoning:
{prev_reasoning or "(none)"}

Numeric outcome of previous action:
{json.dumps(prev_transition, ensure_ascii=False)}

Important:
- The numeric outcome is only a hint.
- The main evidence is the visual difference between Image 1 and Image 2.
- If Image 1 and Image 2 are provided, compare them before choosing an action.
- If the changed region moved after the previous action, that changed region is likely the controlled player.
- If the previous action caused little/no meaningful visual change, do not blindly repeat it.
""".strip()
    else:
        previous_context = """
No previous action is available.
Infer the player and goal from the current image.
""".strip()

    current_meta = {
        "state": meta.get("state"),
        "levels_completed": meta.get("levels_completed"),
        "available_actions": meta.get("available_actions"),
    }
    action6_text = ""
    if action6_candidates:
        action6_text = f"""

ACTION6 coordinate candidates:
{json.dumps(action6_candidates, ensure_ascii=False)}

ACTION6 rule:
- If you choose ACTION6, you must output one of the exact candidate strings from action6_candidates.
- Do not output bare ACTION6.
""".rstrip()
    return f"""
Available actions:
{json.dumps(current_available, ensure_ascii=False)}
{action6_text}

{previous_context}

Image task:
- If two images are provided, Image 1 is BEFORE the previous action and Image 2 is AFTER/current.
- First describe the visual difference.
- Use the visual difference to identify or revise the player hypothesis.
- Then choose the next action.

Rules:
- Return strict JSON only.
- chosen_actions must be a list.
- Output 1 to {config.actions_per_vlm_call} actions.
- Every non-ACTION6 action must be exactly one of Available actions.
- If ACTION6 is used, it must include x and y in the exact candidate string format.
- Do not choose an action before comparing the images when two images are provided.
- Do not rely only on object appearance to decide the player.
- Prefer the object/region that changed after the previous action as the player.
- If the previous action made little/no progress, test a different action or hypothesis.

Current metadata:
{json.dumps(current_meta, ensure_ascii=False)}

Return only:
{{
  "visual_change": "...",
  "player_hypothesis": "...",
  "progress_assessment": "progress / no_progress / uncertain",
  "reasoning": "...",
  "chosen_actions": []
}}
""".strip()
