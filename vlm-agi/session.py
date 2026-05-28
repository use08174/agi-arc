from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from config import AppConfig
from grid import (
    action6_candidates,
    available_action_names,
    is_valid_grid,
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
        "rule_hypotheses": [],
        "last_test_goal": None,
        "last_expected_observation": None,
        "logs": [],
        "closed": False,
        "action_queue": [],
        "current_plan": None,
        "last_action6_candidates": [],
        "last_planned_action_budget": None,
        "action6_history": {},
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
        print(
            json.dumps(
                frame_metadata(raw, game_id=self.config.game_id),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
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
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
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

        if name == "ACTION6" and not payload_text:
            raise ValueError("ACTION6 requires x,y payload")

        if name == "ACTION6" and payload_text:
            allowed_candidates = {
                item["action_text"]
                for item in self.session.get("last_action6_candidates", [])
                if isinstance(item, dict) and item.get("action_text")
            }
            if allowed_candidates and action_text not in allowed_candidates:
                raise ValueError(
                    f"ACTION6 must use one of the suggested candidates: {sorted(allowed_candidates)}"
                )
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

    @staticmethod
    def action6_key(action_text: str) -> str | None:
        if not str(action_text).startswith("ACTION6|"):
            return None
        return str(action_text)

    @staticmethod
    def _compact_rule_hypotheses(plan: dict[str, Any]) -> list[dict[str, str]]:
        hypotheses = plan.get("rule_hypotheses")
        if not isinstance(hypotheses, list):
            return []
        out = []
        for item in hypotheses[:4]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("hypothesis", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "hypothesis": text,
                    "confidence": str(item.get("confidence", "low")).strip() or "low",
                }
            )
        return out

    def update_hypotheses_from_plan(self, plan: dict[str, Any]) -> None:
        self.session["rule_hypotheses"] = self._compact_rule_hypotheses(plan)
        self.session["last_test_goal"] = str(plan.get("best_experiment", "")).strip()
        self.session["last_expected_observation"] = str(
            plan.get("expected_observation", "")
        ).strip()

    def update_hypotheses_from_result(
        self,
        action_text: str,
        transition: dict[str, Any],
        test_result: str,
    ) -> None:
        hypotheses = list(self.session.get("rule_hypotheses", []))
        if not hypotheses:
            return

        note = None
        changed = transition.get("changed_cells")
        if str(action_text).startswith("ACTION6|"):
            if test_result in {"no_visible_effect", "invalid_frame"} or changed == 0:
                note = "Recent evidence: this coordinate click likely has no effect or is invalid."
            elif isinstance(changed, int) and changed > 0:
                note = "Recent evidence: this coordinate click causes a visible local effect."
        elif test_result == "confirmed_progress":
            note = "Recent evidence: this action pattern appears to make real progress."

        if note:
            top = dict(hypotheses[0])
            top["hypothesis"] = f"{top['hypothesis']} {note}".strip()
            hypotheses[0] = top
            self.session["rule_hypotheses"] = hypotheses[:4]

    def get_chosen_actions_from_parsed(
        self, parsed: dict[str, Any], raw_frame: Any, max_actions: int
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
            if len(clean) >= max_actions:
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
        if allowed == ["ACTION6"] or (len(allowed) == 1 and "ACTION6" in allowed):
            candidates = self.build_action6_candidates(raw_frame)
            if candidates:
                return candidates[0]["action_text"]
        non_coordinate = [action for action in allowed if action != "ACTION6"]
        if non_coordinate:
            allowed = non_coordinate
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

    def build_action6_candidates(self, raw_frame: Any) -> list[dict[str, Any]]:
        candidates = []
        current_grid = latest_grid_from_raw(raw_frame)
        candidates.extend(
            action6_candidates(
                current_grid,
                max_candidates=self.config.action6_max_candidates,
                grid_points_per_axis=self.config.action6_grid_points_per_axis,
            )
        )

        transition = self.session.get("last_transition")
        bbox = transition.get("changed_bbox") if isinstance(transition, dict) else None
        if isinstance(bbox, dict):
            x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
            y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
            changed_region = {
                "x": x,
                "y": y,
                "action_text": f"ACTION6|x={x},y={y}",
                "source": "changed_region",
                "detail": "center_of_previous_change",
            }
            existing = {item["action_text"] for item in candidates}
            if changed_region["action_text"] not in existing:
                candidates.insert(0, changed_region)

        history = self.session.get("action6_history", {})
        filtered = []
        for candidate in candidates:
            key = candidate["action_text"]
            stats = history.get(key, {})
            if stats.get("blocked", 0) >= 2 or stats.get("no_effect", 0) >= 2:
                continue
            score = 0
            score += 100 * int(stats.get("confirmed", 0))
            score += 10 * int(stats.get("visible", 0))
            score += 3 * int(stats.get("weak", 0))
            score -= 25 * int(stats.get("blocked", 0))
            score -= 15 * int(stats.get("no_effect", 0))
            score -= 8 * int(stats.get("tries", 0))
            enriched = dict(candidate)
            enriched["score"] = score
            filtered.append(enriched)

        filtered.sort(
            key=lambda item: (
                -int(item.get("score", 0)),
                item.get("source") != "changed_region",
                item.get("source") != "object_center",
            )
        )
        self.session["last_action6_candidates"] = filtered[: self.config.action6_max_candidates]
        return self.session["last_action6_candidates"]

    def adaptive_action_budget(self, raw_frame: Any) -> int:
        if not self.config.adaptive_action_planning:
            return self.config.actions_per_vlm_call

        max_budget = max(1, self.config.max_actions_per_vlm_call)
        base_budget = max(1, min(self.config.actions_per_vlm_call, max_budget))
        logs = [record for record in self.session.get("logs", [])[-3:] if isinstance(record, dict)]
        if not logs:
            return 1

        strengths = []
        recent_action6 = 0
        recent_non_action6 = 0
        repeated_same_action = True
        prev_action = None
        for record in logs:
            action_name = str(record.get("action", ""))
            transition = record.get("transition_numeric") or {}
            changed = transition.get("changed_cells")
            if isinstance(changed, int):
                strengths.append(changed)
            if action_name.startswith("ACTION6"):
                recent_action6 += 1
            else:
                recent_non_action6 += 1
            if prev_action is None:
                prev_action = action_name
            elif action_name != prev_action:
                repeated_same_action = False

        avg_change = sum(strengths) / len(strengths) if strengths else 0.0
        latest_result = str(logs[-1].get("test_result", ""))
        latest_transition = logs[-1].get("transition_numeric") or {}
        latest_changed = latest_transition.get("changed_cells")
        available = available_action_names(raw_frame)
        only_coordinate = bool(available) and set(available) == {"ACTION6"}

        if latest_result in {"invalid_frame", "no_visible_effect"}:
            return 1
        if isinstance(latest_changed, int) and latest_changed == 0:
            return 1
        if latest_result == "weak_or_blocked_effect":
            return 1 if recent_action6 else min(2, base_budget)
        if latest_result == "confirmed_progress":
            if repeated_same_action and recent_non_action6:
                return min(max_budget, 10)
            return min(max_budget, 5)
        if isinstance(latest_changed, int) and latest_changed >= 6:
            if repeated_same_action and recent_non_action6:
                return min(max_budget, 10)
            return min(max_budget, 5)
        if avg_change >= 3:
            return min(max_budget, 3 if recent_action6 or only_coordinate else 5)
        return 1 if recent_action6 or only_coordinate else base_budget

    @staticmethod
    def plan_intent_for_budget(
        max_actions: int,
        available_actions: list[str],
    ) -> str:
        if max_actions >= 8:
            return (
                "You have strong evidence. Prefer a long, consistent action sequence that can repeat a known-good move."
            )
        if max_actions >= 5:
            return (
                "You have some evidence. Prefer a short multi-step plan if the same hypothesis still looks valid."
            )
        if "ACTION6" in available_actions:
            return (
                "Treat ACTION6 as a probe unless visual evidence is strong. Prefer 1 precise coordinate test before committing to a longer plan."
            )
        return "If uncertain, keep the plan short."

    @staticmethod
    def _is_simple_move(action_text: str) -> bool:
        return action_text in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}

    def maybe_expand_action_sequence(
        self,
        action_texts: list[str],
        parsed: dict[str, Any],
        max_actions: int,
    ) -> list[str]:
        if len(action_texts) != 1 or max_actions <= 1:
            return action_texts

        action_text = action_texts[0]
        if not self._is_simple_move(action_text):
            return action_texts

        logs = [
            record for record in self.session.get("logs", [])[-3:] if isinstance(record, dict)
        ]
        if len(logs) < 2:
            return action_texts

        recent_same_move = 0
        for record in reversed(logs):
            if str(record.get("action")) != action_text:
                break
            if str(record.get("test_result")) not in {
                "confirmed_progress",
                "visible_change_no_progress",
            }:
                break
            recent_same_move += 1

        if recent_same_move < 2:
            return action_texts

        progress = str(parsed.get("progress_assessment", "")).strip().lower()
        if progress not in {"progress", "uncertain", ""}:
            return action_texts

        return [action_text] * max_actions

    def ask_vlm_for_action_sequence(
        self, raw_frame: Any
    ) -> tuple[list[str], dict[str, Any], str]:
        images = self.images_for_current_query(raw_frame)
        meta = frame_metadata(raw_frame, game_id=self.config.game_id)
        max_actions = self.adaptive_action_budget(raw_frame)
        candidates = []
        if "ACTION6" in meta.get("available_actions", []):
            candidates = self.build_action6_candidates(raw_frame)
        prompt = build_policy_prompt(
            meta,
            self.session,
            self.config,
            action6_candidates=candidates,
            max_actions=max_actions,
            image_count=len(images),
        )
        prompt += "\n\nPlanning guidance:\n" + self.plan_intent_for_budget(
            max_actions, meta.get("available_actions", [])
        )
        print(f"[vlm] images={len(images)} max_actions={max_actions}")
        raw_text = self.vlm.generate_local_vlm(
            images=images,
            prompt=prompt,
            max_new_tokens=self.config.max_new_tokens,
        )
        parsed = extract_json_object(raw_text)
        self.session["last_vlm_reasoning"] = str(parsed.get("reasoning", "")).strip()
        self.session["last_visual_change"] = str(parsed.get("visual_change", "")).strip()
        self.session["last_player_hypothesis"] = str(
            parsed.get("player_hypothesis", "")
        ).strip()
        self.update_hypotheses_from_plan(parsed)
        self.session["last_planned_action_budget"] = max_actions
        action_texts = self.get_chosen_actions_from_parsed(parsed, raw_frame, max_actions)
        action_texts = self.maybe_expand_action_sequence(action_texts, parsed, max_actions)
        parsed["chosen_actions"] = list(action_texts)
        return action_texts, parsed, raw_text

    def should_clear_action_queue(
        self, raw_before: Any, raw_after: Any, transition: dict[str, Any]
    ) -> tuple[bool, str]:
        old_levels = getattr(raw_before, "levels_completed", None)
        new_levels = getattr(raw_after, "levels_completed", None)
        if (
            self.config.stop_sequence_on_level_change
            and old_levels is not None
            and new_levels is not None
            and old_levels != new_levels
        ):
            return True, "levels_completed changed"
        if self.config.stop_sequence_on_no_change and transition.get("changed_cells") == 0:
            return True, "no visible change"
        if not is_valid_grid(latest_grid_from_raw(raw_after)):
            return True, "invalid frame after action"
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
        old_levels = getattr(raw_before, "levels_completed", None)
        new_levels = getattr(raw_after, "levels_completed", None)
        if old_levels is not None and new_levels is not None and old_levels != new_levels:
            return "confirmed_progress"
        if not is_valid_grid(latest_grid_from_raw(raw_after)):
            return "invalid_frame"
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

        action6_key = self.action6_key(action_text)
        if action6_key is not None:
            history = self.session.setdefault("action6_history", {})
            stats = dict(history.get(action6_key, {}))
            stats["tries"] = int(stats.get("tries", 0)) + 1
            if test_result == "confirmed_progress":
                stats["confirmed"] = int(stats.get("confirmed", 0)) + 1
            elif test_result == "visible_change_no_progress":
                stats["visible"] = int(stats.get("visible", 0)) + 1
            elif test_result == "weak_or_blocked_effect":
                stats["weak"] = int(stats.get("weak", 0)) + 1
                if transition.get("changed_cells") == 0:
                    stats["no_effect"] = int(stats.get("no_effect", 0)) + 1
            elif test_result == "no_visible_effect":
                stats["no_effect"] = int(stats.get("no_effect", 0)) + 1
            elif test_result == "invalid_frame":
                stats["blocked"] = int(stats.get("blocked", 0)) + 1
            history[action6_key] = stats

        self.update_hypotheses_from_result(action_text, transition, test_result)

        record = {
            "step": len(self.session["logs"]),
            "action": action_text,
            "vlm_called": vlm_called,
            "planned_action_budget": self.session.get("last_planned_action_budget"),
            "plan": plan,
            "rule_hypotheses": list(self.session.get("rule_hypotheses", [])),
            "test_goal": self.session.get("last_test_goal"),
            "expected_observation": self.session.get("last_expected_observation"),
            "planned_actions": plan.get("chosen_actions") if isinstance(plan, dict) else None,
            "queue_before": queue_before,
            "queue_after": list(self.session.get("action_queue", [])),
            "queue_cleared": clear_queue,
            "queue_clear_reason": clear_reason,
            "transition_numeric": transition,
            "before_meta": frame_metadata(raw, game_id=self.config.game_id),
            "after_meta": frame_metadata(after_raw, game_id=self.config.game_id),
            "raw_vlm_output": raw_vlm_output if vlm_called else "",
            "test_result": test_result,
        }

        self.session["prev_grid"] = before_grid
        self.session["last_action"] = action_text
        self.session["last_transition"] = transition
        self.session["logs"].append(record)
        self.session["raw"] = after_raw

        with Path(self.config.log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        reasoning = plan.get("reasoning") if isinstance(plan, dict) else ""
        test_goal = self.session.get("last_test_goal") or ""
        print(f"[step {record['step']}]")
        if test_goal:
            print(f"test: {test_goal}")
        if reasoning:
            print(reasoning)
        print(f"action: {action_text}")
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

        print(f"logs: {self.config.log_path}")
        if close_at_end:
            self.close_scorecard()
        return logs
