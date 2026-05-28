from __future__ import annotations

import json
import re
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np

from config import AppConfig
from grid import (
    available_action_names,
    display_grid,
    frame_metadata,
    grid_to_rgb_array,
    latest_grid_from_raw,
    as_list_grid,
)
from model import VLMManager, extract_json_object
from prompts import build_policy_prompt


def build_default_session_state() -> dict[str, Any]:
    return {
        "arcade": None,
        "wrapper": None,
        "scorecard_id": None,
        "raw": None,
        "prev_grid": None,
        "last_action": None,
        "last_transition": None,
        "last_vlm_reasoning": None,
        "last_visual_change": None,
        "last_player_hypothesis": None,
        "logs": [],
        "closed": False,
        "action_queue": [],
        "current_plan": None,
    }


class VLMArcRunner:
    def __init__(self, config: AppConfig, vlm: VLMManager):
        self.config = config
        self.vlm = vlm
        self.session = build_default_session_state()

    def start_arc_session(self, *, display_initial: bool = True) -> dict[str, Any]:
        from arc_agi import Arcade, OperationMode

        mode = self.config.arc_mode.lower()
        if mode in {"online", "competition"} and not self.config.arc_api_key:
            raise ValueError("ARC_API_KEY is required for online/competition mode.")

        arcade = Arcade(
            arc_api_key=self.config.arc_api_key or "test-key-123",
            arc_base_url=self.config.arc_base_url,
            operation_mode=OperationMode(mode),
            environments_dir=self.config.environments_dir,
            recordings_dir=self.config.recordings_dir,
        )
        scorecard_id = arcade.create_scorecard(
            source_url="vlm-policy-multi-action-visual-transition",
            tags=["vlm-only", "qwen3-vl", "multi-action", "visual-transition"],
        )
        wrapper = arcade.make(
            self.config.game_id,
            scorecard_id=scorecard_id,
            save_recording=self.config.save_recording,
            include_frame_data=True,
            render_mode=None,
        )
        raw = wrapper.reset()

        self.session = build_default_session_state()
        self.session.update(
            {
                "arcade": arcade,
                "wrapper": wrapper,
                "scorecard_id": scorecard_id,
                "raw": raw,
            }
        )

        print("arc_mode:", mode)
        print("scorecard_id:", scorecard_id)
        print(
            "scorecard_url:",
            f"{self.config.arc_base_url.rstrip('/')}/scorecards/{scorecard_id}",
        )
        print("initial_meta:")
        pprint(frame_metadata(raw, game_id=self.config.game_id))
        if display_initial:
            display_grid(latest_grid_from_raw(raw), title="initial frame")
        return self.session

    def close_scorecard(self) -> dict[str, Any] | None:
        if self.session.get("closed"):
            print("scorecard already closed")
            return None

        arcade = self.session.get("arcade")
        scorecard_id = self.session.get("scorecard_id")
        if arcade is None or scorecard_id is None:
            print("no active scorecard")
            return None

        try:
            summary = arcade.close_scorecard(scorecard_id)
        except Exception as exc:
            print("close_scorecard failed:", repr(exc))
            summary = None

        self.session["closed"] = True
        print("scorecard_id:", scorecard_id)
        print(
            "scorecard_url:",
            f"{self.config.arc_base_url.rstrip('/')}/scorecards/{scorecard_id}",
        )

        if summary is not None:
            Path(self.config.summary_path).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print("wrote summary:", self.config.summary_path)
            pprint(summary)
        return summary

    @staticmethod
    def normalize_chosen_action_text(action_text: str) -> str:
        action_text = str(action_text).strip()
        if not action_text:
            raise ValueError("empty action")
        if re.fullmatch(r"[0-9]+", action_text):
            return f"ACTION{action_text}"
        if re.match(r"^[0-9]+\|", action_text):
            number, rest = action_text.split("|", 1)
            return f"ACTION{number}|{rest}"
        if match := re.match(r"(?i)^action\s*([0-9]+)(.*)$", action_text):
            return f"ACTION{match.group(1)}{match.group(2)}".replace(" ", "")
        return action_text

    def validate_action_text(self, action_text: str, raw_frame: Any) -> str:
        action_text = self.normalize_chosen_action_text(action_text)
        name, payload_text = (
            action_text.split("|", 1) if "|" in action_text else (action_text, "")
        )
        allowed = available_action_names(raw_frame)
        if name not in allowed:
            raise ValueError(f"{name} not in available_actions={allowed}")

        if name == "ACTION6" and payload_text:
            arr = np.asarray(latest_grid_from_raw(raw_frame))
            h, w = arr.shape[:2]
            parts = {}
            for item in payload_text.split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                parts[key.strip()] = int(value)
            x = parts.get("x")
            y = parts.get("y")
            if x is None or y is None:
                raise ValueError("ACTION6 payload must include x and y")
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(
                    f"ACTION6 coordinate out of bounds: x={x}, y={y}, grid={w}x{h}"
                )
        return action_text

    def parse_action(self, action_text: str) -> tuple[Any, dict[str, int] | None]:
        from arcengine import GameAction

        action_text = self.normalize_chosen_action_text(action_text)
        if "|" not in action_text:
            return GameAction.from_name(action_text), None

        name, payload_text = action_text.split("|", 1)
        payload: dict[str, int] = {}
        for part in payload_text.split(","):
            if not part.strip():
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = int(value)
        return GameAction.from_name(name), payload

    def get_chosen_actions_from_parsed(
        self, parsed: dict[str, Any], raw_frame: Any
    ) -> list[str]:
        if not isinstance(parsed, dict):
            raise ValueError(f"parsed output is not dict: {parsed}")

        actions = parsed.get("chosen_actions")
        if actions is None and parsed.get("chosen_action"):
            actions = [parsed.get("chosen_action")]
        if actions is None:
            decision = parsed.get("decision")
            if isinstance(decision, dict) and decision.get("chosen_action"):
                actions = [decision.get("chosen_action")]
        if isinstance(actions, str):
            actions = [actions]
        if not isinstance(actions, list):
            raise ValueError(f"No chosen_actions found in parsed output: {parsed}")

        clean = []
        for action in actions:
            if len(clean) >= self.config.actions_per_vlm_call:
                break
            clean.append(self.validate_action_text(action, raw_frame))
        if not clean:
            raise ValueError(f"Empty chosen_actions after validation: {parsed}")
        return clean

    @staticmethod
    def summarize_transition(
        before_grid: Any,
        after_grid: Any,
        *,
        after_raw: Any | None = None,
    ) -> dict[str, Any]:
        before = np.asarray(as_list_grid(before_grid), dtype=int)
        after = np.asarray(as_list_grid(after_grid), dtype=int)
        out = {
            "changed_cells": None,
            "changed_bbox": None,
            "color_changes": [],
            "state": None,
            "levels_completed": None,
        }

        if after_raw is not None:
            state = getattr(after_raw, "state", None)
            out["state"] = getattr(state, "name", str(state)) if state is not None else None
            out["levels_completed"] = getattr(after_raw, "levels_completed", None)

        if before.shape != after.shape or not before.size:
            out["changed_cells"] = "shape_changed"
            return out

        diff = before != after
        out["changed_cells"] = int(diff.sum())
        if diff.any():
            ys, xs = np.where(diff)
            out["changed_bbox"] = {
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
            }
            pairs: dict[tuple[int, int], int] = {}
            for before_color, after_color in zip(before[diff].tolist(), after[diff].tolist()):
                key = (int(before_color), int(after_color))
                pairs[key] = pairs.get(key, 0) + 1
            out["color_changes"] = [
                {"from": int(key[0]), "to": int(key[1]), "count": int(value)}
                for key, value in sorted(pairs.items(), key=lambda item: -item[1])[:8]
            ]
        return out

    def fallback_action(self, raw_frame: Any) -> str:
        allowed = available_action_names(raw_frame)
        if not allowed:
            raise ValueError("no available actions")
        recent = [
            record.get("action")
            for record in self.session.get("logs", [])[-6:]
            if isinstance(record, dict)
        ]
        for action in allowed:
            if action not in recent:
                return action
        return allowed[0]

    def images_for_current_query(self, raw_frame: Any) -> list[np.ndarray]:
        current_grid = latest_grid_from_raw(raw_frame)
        current_image = grid_to_rgb_array(
            current_grid,
            scale=self.config.image_scale,
            draw_grid=self.config.draw_grid,
        )
        prev_grid = self.session.get("prev_grid")
        last_action = self.session.get("last_action")
        if prev_grid is not None and last_action is not None:
            prev_image = grid_to_rgb_array(
                prev_grid,
                scale=self.config.image_scale,
                draw_grid=self.config.draw_grid,
            )
            return [prev_image, current_image]
        return [current_image]

    def ask_vlm_for_action_sequence(
        self, raw_frame: Any
    ) -> tuple[list[str], dict[str, Any], str]:
        images = self.images_for_current_query(raw_frame)
        meta = frame_metadata(raw_frame, game_id=self.config.game_id)
        prompt = build_policy_prompt(meta, self.session, self.config)
        raw_text = self.vlm.generate_local_vlm(
            images=images,
            prompt=prompt,
            max_new_tokens=self.config.max_new_tokens,
        )
        if self.config.print_vlm_output:
            print("RAW VLM OUTPUT:")
            print(raw_text)
        parsed = extract_json_object(raw_text)
        self.session["last_vlm_reasoning"] = str(parsed.get("reasoning", "")).strip()
        self.session["last_visual_change"] = str(parsed.get("visual_change", "")).strip()
        self.session["last_player_hypothesis"] = str(
            parsed.get("player_hypothesis", "")
        ).strip()
        action_texts = self.get_chosen_actions_from_parsed(parsed, raw_frame)
        return action_texts, parsed, raw_text

    def should_clear_action_queue(
        self, raw_before: Any, raw_after: Any, transition: dict[str, Any]
    ) -> tuple[bool, str]:
        old_levels = getattr(raw_before, "levels_completed", None)
        new_levels = getattr(raw_after, "levels_completed", None)
        if self.config.stop_sequence_on_level_change and old_levels != new_levels:
            return True, "levels_completed changed"
        if self.config.stop_sequence_on_no_change and transition.get("changed_cells") == 0:
            return True, "no visible change"
        if self.config.stop_sequence_on_actions_change:
            if available_action_names(raw_before) != available_action_names(raw_after):
                return True, "available_actions changed"
        return False, ""

    @staticmethod
    def infer_test_result(
        raw_before: Any,
        raw_after: Any,
        transition: dict[str, Any],
    ) -> str:
        if getattr(raw_before, "levels_completed", None) != getattr(
            raw_after, "levels_completed", None
        ):
            return "confirmed_progress"
        if transition.get("changed_cells") == 0:
            return "no_visible_effect"
        if transition.get("changed_cells") <= 2:
            return "weak_or_blocked_effect"
        return "visible_change_no_progress"

    def step_once(self, *, display_after: bool = True) -> dict[str, Any]:
        raw = self.session.get("raw")
        wrapper = self.session.get("wrapper")
        if raw is None or wrapper is None:
            raise RuntimeError("No active session. Run start_arc_session() first.")

        before_grid = latest_grid_from_raw(raw)
        vlm_called = False
        raw_vlm_output = ""
        queue_before = list(self.session.get("action_queue", []))

        if not self.session.get("action_queue"):
            try:
                action_texts, plan, raw_vlm_output = self.ask_vlm_for_action_sequence(raw)
                self.session["action_queue"] = list(action_texts)
                self.session["current_plan"] = plan
                vlm_called = True
            except Exception as exc:
                print("VLM action parse/validation failed:", repr(exc))
                fallback = self.fallback_action(raw)
                plan = {
                    "reasoning": f"fallback due to error: {repr(exc)}",
                    "chosen_actions": [fallback],
                }
                self.session["action_queue"] = [fallback]
                self.session["current_plan"] = plan
                vlm_called = True
        else:
            plan = self.session.get("current_plan") or {
                "reasoning": "continuing queued VLM action sequence",
                "chosen_actions": list(self.session.get("action_queue", [])),
            }

        action_text = self.session["action_queue"].pop(0)
        action, payload = self.parse_action(action_text)
        after_raw = wrapper.step(action, data=payload)
        after_grid = latest_grid_from_raw(after_raw)
        transition = self.summarize_transition(
            before_grid, after_grid, after_raw=after_raw
        )
        test_result = self.infer_test_result(raw, after_raw, transition)

        clear_queue, clear_reason = self.should_clear_action_queue(
            raw, after_raw, transition
        )
        if clear_queue:
            self.session["action_queue"] = []

        record = {
            "step": len(self.session["logs"]),
            "action": action_text,
            "vlm_called": vlm_called,
            "plan": plan,
            "planned_actions": plan.get("chosen_actions") if isinstance(plan, dict) else None,
            "queue_before": queue_before,
            "queue_after": list(self.session.get("action_queue", [])),
            "queue_cleared": clear_queue,
            "queue_clear_reason": clear_reason,
            "transition_numeric": transition,
            "before_meta": frame_metadata(raw, game_id=self.config.game_id),
            "after_meta": frame_metadata(after_raw, game_id=self.config.game_id),
            "raw_vlm_output": raw_vlm_output if vlm_called else "",
            "test_goal": plan.get("test_goal") if isinstance(plan, dict) else None,
            "test_result": test_result,
        }

        self.session["prev_grid"] = before_grid
        self.session["last_action"] = action_text
        self.session["last_transition"] = transition
        self.session["logs"].append(record)
        self.session["raw"] = after_raw

        with Path(self.config.log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        print("=" * 80)
        print("step:", record["step"])
        print("vlm_called:", vlm_called)
        print("action:", action_text)
        print("planned_actions:", record["planned_actions"])
        print("queue_after:", record["queue_after"])
        if clear_queue:
            print("queue_cleared:", clear_reason)
        print("reasoning:", plan.get("reasoning") if isinstance(plan, dict) else None)
        print("numeric transition:")
        pprint(transition)
        if display_after:
            display_grid(after_grid, title=f"after {action_text}")
        return record

    @staticmethod
    def is_terminal_raw(raw_frame: Any) -> bool:
        state = getattr(raw_frame, "state", None)
        state_name = getattr(state, "name", str(state)) if state is not None else ""
        state_name = str(state_name).upper()
        return state_name in {"WIN", "GAME_OVER", "DONE"} or bool(
            getattr(raw_frame, "won", False)
        )

    def run_episode(
        self,
        *,
        max_steps: int | None = None,
        close_at_end: bool = False,
        display_after: bool = True,
    ) -> list[dict[str, Any]]:
        logs = []
        limit = int(max_steps or self.config.max_steps)
        for _ in range(limit):
            record = self.step_once(display_after=display_after)
            logs.append(record)
            if self.is_terminal_raw(self.session["raw"]):
                print("terminal state reached")
                break

        print("wrote logs:", self.config.log_path)
        print("remaining action_queue:", self.session.get("action_queue", []))
        if close_at_end:
            self.close_scorecard()
        return logs
