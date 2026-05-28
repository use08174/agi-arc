from __future__ import annotations

import json
from typing import Any

from config import AppConfig
from grid import frame_metadata, latest_grid_from_raw, summarize_objects, summarize_spatial_layout


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
Your main job is not just to act. Your main job is to form and test rules about the game.

Controls:
ACTION1=up
ACTION2=down
ACTION3=left
ACTION4=right
ACTION5=interact/special
ACTION6=x,y coordinate
ACTION7=extra

Hard control semantics:
- Treat ACTION1, ACTION2, ACTION3, ACTION4 as fixed symbolic controls.
- Do not reinterpret them from the image.
- ACTION1 means only up.
- ACTION2 means only down.
- ACTION3 means only left.
- ACTION4 means only right.
- These movement semantics are known in advance and are not hypotheses to test.
- Never use a turn just to verify whether ACTION1-ACTION4 mean up/down/left/right.
- Never describe ACTION4 as "right and up", "diagonal", "toward a platform", or any other composite move unless the environment response after pressing it causes that outcome.
- Separate control input from environment response:
  - control input = the commanded direction above
  - environment response = how the world changed after that input
- When comparing before/after images, describe the observed response, but do not redefine the meaning of the action itself.

Do not describe the image as a character, animal, face, body, whale, fish, or creature.

Return strict JSON only:
{
  "visual_change": "what changed from Image 1 to Image 2, or 'only one image' if no previous image",
  "player_hypothesis": "which object is likely controlled and why",
  "strategic_goal": "what winning probably requires overall",
  "immediate_goal": "what the next short subgoal should accomplish",
  "rule_hypotheses": [
    {"hypothesis": "...", "confidence": "low/medium/high"}
  ],
  "best_experiment": "what rule this action is testing",
  "expected_observation": "what result would support or weaken the rule hypothesis",
  "plan_summary": "2-4 step plan toward the immediate goal",
  "plan_stop_condition": "what observation should make you stop or revise the plan",
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
    max_actions: int | None = None,
    image_count: int | None = None,
) -> str:
    planned_actions = int(max_actions or config.actions_per_vlm_call)
    min_actions = min(3, planned_actions)
    current_available = list(meta.get("available_actions", []))
    unavailable = [
        action for action in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
        if action not in current_available
    ]
    prev_reasoning = session.get("last_vlm_reasoning")
    prev_visual_change = session.get("last_visual_change")
    prev_player_hypothesis = session.get("last_player_hypothesis")
    prev_rule_hypotheses = session.get("rule_hypotheses") or []
    prev_test_goal = session.get("last_test_goal")
    prev_expected_observation = session.get("last_expected_observation")
    prev_strategic_goal = session.get("last_strategic_goal")
    prev_immediate_goal = session.get("last_immediate_goal")
    prev_plan_summary = session.get("last_plan_summary")
    prev_plan_stop_condition = session.get("last_plan_stop_condition")
    learned_action_meanings = session.get("learned_action_meanings") or {}
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

Previous rule hypotheses:
{json.dumps(prev_rule_hypotheses, ensure_ascii=False)}

Previous strategic goal:
{prev_strategic_goal or "(none)"}

Previous immediate goal:
{prev_immediate_goal or "(none)"}

Previous plan summary:
{prev_plan_summary or "(none)"}

Previous plan stop condition:
{prev_plan_stop_condition or "(none)"}

Previous test goal:
{prev_test_goal or "(none)"}

Previous expected observation:
{prev_expected_observation or "(none)"}

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
    current_grid = latest_grid_from_raw(session.get("raw_for_prompt")) if session.get("raw_for_prompt") is not None else []
    detected_objects = summarize_objects(current_grid, max_objects=12) if current_grid else []
    spatial_layout = summarize_spatial_layout(current_grid) if current_grid else {}
    image_count_text = ""
    if image_count is not None:
        if image_count >= 2:
            image_count_text = f"""

Input confirmation:
- You are receiving exactly {image_count} images now.
- Image 1 is BEFORE the previous action.
- Image 2 is AFTER/current.
- You must compare them before choosing the next action.
""".rstrip()
        else:
            image_count_text = f"""

Input confirmation:
- You are receiving exactly {image_count} image now.
- There is no previous image to compare against.
""".rstrip()
    action6_text = ""
    if action6_candidates:
        action6_text = f"""

ACTION6 coordinate candidates:
{json.dumps(action6_candidates, ensure_ascii=False)}

ACTION6 rule:
- If you choose ACTION6, you must output one of the exact candidate strings from action6_candidates.
- Do not output bare ACTION6.
- Do not invent new x,y coordinates.
- If movement actions ACTION1-ACTION4 are available and the control rule is still uncertain, prefer testing movement before ACTION6.
""".rstrip()
    action_menu = []
    action_meanings = {
        "ACTION1": "move up",
        "ACTION2": "move down",
        "ACTION3": "move left",
        "ACTION4": "move right",
        "ACTION5": "interact/special",
        "ACTION6": "coordinate action",
        "ACTION7": "extra action",
    }
    for action in current_available:
        action_menu.append({"action": action, "meaning": action_meanings.get(action, "unknown")})
    return f"""
Available actions:
{json.dumps(current_available, ensure_ascii=False)}
Unavailable actions for this state:
{json.dumps(unavailable, ensure_ascii=False)}
Action menu for this exact state:
{json.dumps(action_menu, ensure_ascii=False)}
{image_count_text}
{action6_text}

Learned control/response evidence:
{json.dumps(learned_action_meanings, ensure_ascii=False)}

Detected connected components:
{json.dumps(detected_objects, ensure_ascii=False)}

Spatial layout summary:
{json.dumps(spatial_layout, ensure_ascii=False)}

{previous_context}

Image task:
- If two images are provided, Image 1 is BEFORE the previous action and Image 2 is AFTER/current.
- First describe the visual difference.
- Use the visual difference to identify or revise the player hypothesis.
- Then choose the next action.

Rules:
- Return strict JSON only.
- chosen_actions must be a list.
- Output {min_actions} to {planned_actions} actions.
- Every non-ACTION6 action must be exactly one of Available actions.
- Available actions are a hard constraint for the current state.
- If an action is not listed in Available actions, it is currently impossible and must not appear in chosen_actions, plan_summary, or best_experiment.
- If ACTION6 is used, it must include x and y in the exact candidate string format.
- Do not choose an action before comparing the images when two images are provided.
- Do not rely only on object appearance to decide the player.
- Prefer the object/region that changed after the previous action as the player.
- If the previous action made little/no progress, test a different action or hypothesis.
- Avoid ACTION6 early unless the screen strongly suggests point-and-click interaction or ACTION1-ACTION4 are unavailable.
- If movement semantics are already supported by prior evidence, stop re-testing them unless that test serves a concrete subgoal.
- Do not take actions whose only purpose is "see what happens" if you already have a stronger goal-directed plan.
- Do not write best_experiment like "test whether ACTION3 moves left" or "test whether ACTION4 moves right". Those are already known.
- Use ACTION1-ACTION4 to pursue spatial goals, not to rediscover the control mapping.
- Use the connected components and spatial layout summary to infer corridors, blockers, targets, and likely routes.
- If upward progress seems desirable but ACTION1 is unavailable, first choose an available repositioning action such as ACTION3, ACTION4, ACTION7, or ACTION6 if those are available.
- If downward progress seems desirable but ACTION2 is unavailable, first choose an available repositioning action instead of naming ACTION2.
- Before writing chosen_actions, mentally check each action against the Action menu for this exact state.
- chosen_actions should be the direct executable translation of your plan, not an abstract wish like "go up" when up is unavailable.

Current metadata:
{json.dumps(current_meta, ensure_ascii=False)}

Current objective:
- Maintain a small set of explicit rule hypotheses.
- Infer an explicit likely win condition and name it as strategic_goal.
- Choose an immediate_goal that advances the strategic_goal.
- Choose actions that either advance the immediate_goal or distinguish between competing rule hypotheses.
- Prefer experiments that produce informative differences, not just movement.
- Convert rule understanding into a concrete short plan instead of re-probing the same mechanic.
- Always provide at least 3 actions when possible.
- If the rule already looks stable and a repeated move is the best test, fill chosen_actions with a longer repeated sequence instead of returning only one action.
- If a repeated move is justified, prefer emitting multiple actions at once up to the allowed budget.

Return only:
{{
  "visual_change": "...",
  "player_hypothesis": "...",
  "strategic_goal": "...",
  "immediate_goal": "...",
  "rule_hypotheses": [
    {{"hypothesis": "...", "confidence": "low/medium/high"}}
  ],
  "best_experiment": "...",
  "expected_observation": "...",
  "plan_summary": "...",
  "plan_stop_condition": "...",
  "progress_assessment": "progress / no_progress / uncertain",
  "reasoning": "...",
  "chosen_actions": []
}}
""".strip()
