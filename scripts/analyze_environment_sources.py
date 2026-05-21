#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"\b(?:RESET|ACTION[1-7])\b")
HINT_KEYWORDS = {
    "alignment": (
        "align",
        "overlap",
        "same_position",
        "distance",
        "target",
        "match",
        "goal",
    ),
    "painting": (
        "paint",
        "fill",
        "color",
        "cell",
        "brush",
        "draw",
    ),
    "navigation": (
        "move",
        "wall",
        "collision",
        "collide",
        "player",
        "path",
        "position",
    ),
    "transformation": (
        "rotate",
        "rotation",
        "flip",
        "mirror",
        "transform",
        "copy",
        "scale",
    ),
    "selection": (
        "select",
        "selected",
        "cursor",
        "click",
        "actioninput",
        "input",
    ),
    "physics": (
        "push",
        "velocity",
        "gravity",
        "vx",
        "vy",
        "bounce",
    ),
}
GOAL_RE = re.compile(r"\b(?:WIN|GAME_OVER|NOT_FINISHED|won|win|lose|game_over|next_level|complete|success|fail)\b", re.IGNORECASE)
OBFUSCATED_NAME_RE = re.compile(r"^[a-z]{9,}$")


@dataclass(slots=True)
class SpriteSummary:
    name: str
    width: int | None = None
    height: int | None = None
    colors: list[int] = field(default_factory=list)
    visible: bool | None = None
    collidable: bool | None = None


@dataclass(slots=True)
class GameSourceCard:
    game_id: str
    path: str
    parse_ok: bool
    error: str | None = None
    levels: int = 0
    grid_sizes: list[list[int]] = field(default_factory=list)
    sprites: list[SpriteSummary] = field(default_factory=list)
    level_sprite_counts: list[int] = field(default_factory=list)
    initial_positions: list[list[int]] = field(default_factory=list)
    action_refs: list[str] = field(default_factory=list)
    uses_action6: bool = False
    uses_action7: bool = False
    uses_action_input: bool = False
    control_family: str = "unknown"
    grid_bucket: str = "unknown"
    recommended_strategy: str = "unknown"
    class_names: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    broad_source_hints: list[str] = field(default_factory=list)
    action_context_hints: list[str] = field(default_factory=list)
    goal_context_hints: list[str] = field(default_factory=list)
    task_family_prior: str = "unknown"
    confidence: float = 0.0


def literal_eval_safe(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ""


def keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def nested_ints(value: Any) -> list[int]:
    out: list[int] = []
    if isinstance(value, int):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(nested_ints(item))
    return out


class EnvironmentSourceAnalyzer(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.source_lower = source.lower()
        self.levels = 0
        self.grid_sizes: list[list[int]] = []
        self.sprites: list[SpriteSummary] = []
        self.level_sprite_counts: list[int] = []
        self.initial_positions: list[list[int]] = []
        self.class_names: list[str] = []
        self.function_names: list[str] = []
        self.uses_action_input = False
        self.action_contexts: list[str] = []
        self.goal_contexts: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        if any(call_name(base) == "ARCBaseGame" for base in node.bases):
            self.source_lower += " arcbasegame"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_names.append(node.name)
        self._record_focused_context(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_names.append(node.name)
        self._record_focused_context(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "ActionInput":
            self.uses_action_input = True

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in {"x", "y"} and isinstance(node.value, ast.Name) and node.value.id.lower().endswith("input"):
            self.uses_action_input = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node.func)
        if name == "Level":
            self._record_level(node)
        elif name == "Sprite":
            self._record_sprite(node)
        elif name == "set_position":
            self._record_position(node)
        self.generic_visit(node)

    def _record_level(self, node: ast.Call) -> None:
        self.levels += 1
        grid_size = keyword_value(node, "grid_size")
        grid_value = literal_eval_safe(grid_size) if grid_size is not None else None
        if (
            isinstance(grid_value, (list, tuple))
            and len(grid_value) == 2
            and all(isinstance(item, int) for item in grid_value)
        ):
            self.grid_sizes.append([int(grid_value[0]), int(grid_value[1])])
        sprites = keyword_value(node, "sprites")
        if isinstance(sprites, ast.List):
            self.level_sprite_counts.append(len(sprites.elts))

    def _record_sprite(self, node: ast.Call) -> None:
        pixels = keyword_value(node, "pixels")
        name_node = keyword_value(node, "name")
        visible = keyword_value(node, "visible")
        collidable = keyword_value(node, "collidable")
        pixel_value = literal_eval_safe(pixels) if pixels is not None else None
        sprite_name = literal_eval_safe(name_node) if name_node is not None else None
        colors = sorted(set(nested_ints(pixel_value)))
        height = len(pixel_value) if isinstance(pixel_value, list) else None
        width = max((len(row) for row in pixel_value if isinstance(row, list)), default=0) if isinstance(pixel_value, list) else None
        self.sprites.append(
            SpriteSummary(
                name=str(sprite_name or f"sprite-{len(self.sprites) + 1}"),
                width=width or None,
                height=height,
                colors=[int(color) for color in colors],
                visible=literal_eval_safe(visible) if visible is not None else None,
                collidable=literal_eval_safe(collidable) if collidable is not None else None,
            )
        )

    def _record_position(self, node: ast.Call) -> None:
        if len(node.args) < 2:
            return
        x = literal_eval_safe(node.args[0])
        y = literal_eval_safe(node.args[1])
        if isinstance(x, int) and isinstance(y, int):
            self.initial_positions.append([x, y])

    def build_card(self) -> GameSourceCard:
        action_refs = sorted(set(ACTION_RE.findall(self.source)))
        broad_hints = self._source_hints(self.source_lower)
        action_context_text = "\n".join(self.action_contexts).lower()
        goal_context_text = "\n".join(self.goal_contexts).lower()
        action_hints = self._source_hints(action_context_text)
        goal_hints = self._source_hints(goal_context_text)
        hints = sorted(set(action_hints + goal_hints))
        control_family = infer_control_family(action_refs)
        grid_bucket = infer_grid_bucket(self.grid_sizes)
        recommended_strategy = infer_recommended_strategy(control_family, grid_bucket)
        prior, confidence = infer_task_family(
            hints=hints,
            action_refs=action_refs,
            uses_action_input=self.uses_action_input,
            sprite_count=len(self.sprites),
            grid_sizes=self.grid_sizes,
            control_family=control_family,
            grid_bucket=grid_bucket,
        )
        return GameSourceCard(
            game_id=infer_game_id(self.path),
            path=str(self.path),
            parse_ok=True,
            levels=self.levels,
            grid_sizes=unique_lists(self.grid_sizes),
            sprites=self.sprites,
            level_sprite_counts=self.level_sprite_counts,
            initial_positions=unique_lists(self.initial_positions),
            action_refs=action_refs,
            uses_action6="ACTION6" in action_refs,
            uses_action7="ACTION7" in action_refs,
            uses_action_input=self.uses_action_input,
            control_family=control_family,
            grid_bucket=grid_bucket,
            recommended_strategy=recommended_strategy,
            class_names=sorted(set(self.class_names)),
            function_names=sorted(set(self._non_obfuscated_names(self.function_names))),
            source_hints=hints,
            broad_source_hints=broad_hints,
            action_context_hints=action_hints,
            goal_context_hints=goal_hints,
            task_family_prior=prior,
            confidence=confidence,
        )

    def _source_hints(self, source_lower: str) -> list[str]:
        hints: list[str] = []
        for family, keywords in HINT_KEYWORDS.items():
            if any(keyword in source_lower for keyword in keywords):
                hints.append(family)
        return hints

    def _record_focused_context(self, node: ast.AST) -> None:
        segment = ast.get_source_segment(self.source, node) or ""
        if ACTION_RE.search(segment):
            self.action_contexts.append(segment)
        if GOAL_RE.search(segment):
            self.goal_contexts.append(segment)

    def _non_obfuscated_names(self, names: list[str]) -> list[str]:
        return [name for name in names if not OBFUSCATED_NAME_RE.fullmatch(name)]


def infer_game_id(path: Path) -> str:
    for part in reversed(path.parts):
        stem = Path(part).stem
        if re.fullmatch(r"[a-z]{2}\d{2}", stem):
            return stem
    return path.stem


def unique_lists(items: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for item in items:
        key = tuple(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def infer_control_family(action_refs: list[str]) -> str:
    actions = set(action_refs)
    simple_moves = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    has_move_set = bool(simple_moves & actions)
    has_coord = "ACTION6" in actions
    extra_simple = bool({"ACTION5", "ACTION7"} & actions)
    if has_coord and not has_move_set and not extra_simple:
        return "coordinate_only"
    if has_coord and has_move_set:
        return "mixed_move_coordinate"
    if has_coord and extra_simple:
        return "coordinate_mode"
    if has_move_set and not has_coord and not extra_simple:
        return "movement_only"
    if has_move_set and extra_simple and not has_coord:
        return "movement_mode"
    if extra_simple and not has_coord and not has_move_set:
        return "mode_only"
    return "unknown"


def infer_grid_bucket(grid_sizes: list[list[int]]) -> str:
    if not grid_sizes:
        return "unknown"
    max_dim = max(max(size) for size in grid_sizes if size)
    varied = len(unique_lists(grid_sizes)) > 1
    if max_dim <= 12:
        bucket = "small"
    elif max_dim <= 32:
        bucket = "medium"
    else:
        bucket = "large"
    return f"{bucket}_varied" if varied else bucket


def infer_recommended_strategy(control_family: str, grid_bucket: str) -> str:
    large = grid_bucket.startswith("large")
    small = grid_bucket.startswith("small")
    if control_family == "coordinate_only":
        return "object_coordinate_probe" if large else "dense_coordinate_probe"
    if control_family in {"mixed_move_coordinate", "coordinate_mode"}:
        return "learn_simple_actions_then_object_clicks"
    if control_family == "movement_only":
        return "state_graph_bfs" if small else "movement_axis_model"
    if control_family == "movement_mode":
        return "learn_mode_then_movement_model"
    if control_family == "mode_only":
        return "mode_cycle_probe"
    return "generic_refinement_probe"


def infer_task_family(
    *,
    hints: list[str],
    action_refs: list[str],
    uses_action_input: bool,
    sprite_count: int,
    grid_sizes: list[list[int]],
    control_family: str,
    grid_bucket: str,
) -> tuple[str, float]:
    scores = {
        "coordinate_interaction": 0.0,
        "navigation": 0.0,
        "hybrid": 0.0,
        "mode_control": 0.0,
        "transformation": 0.0,
        "alignment": 0.0,
        "unknown": 0.0,
    }
    if control_family == "coordinate_only":
        scores["coordinate_interaction"] += 2.0
    elif control_family in {"mixed_move_coordinate", "coordinate_mode"}:
        scores["hybrid"] += 2.0
    elif control_family in {"movement_only", "movement_mode"}:
        scores["navigation"] += 1.8
    elif control_family == "mode_only":
        scores["mode_control"] += 1.5
    for hint in hints:
        if hint == "painting" or hint == "selection":
            scores["coordinate_interaction"] += 0.5
        elif hint == "physics":
            scores["navigation"] += 0.3
        elif hint in scores:
            scores[hint] += 0.5
    if "ACTION6" in action_refs or uses_action_input:
        scores["coordinate_interaction"] += 0.4
    if {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} & set(action_refs):
        scores["navigation"] += 0.3
    if sprite_count <= 2 and grid_sizes:
        max_dim = max(max(size) for size in grid_sizes if size)
        if max_dim <= 12:
            scores["navigation"] += 0.4
    if grid_bucket.startswith("large") and control_family == "coordinate_only":
        scores["coordinate_interaction"] += 0.4
    best_family, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "unknown", 0.0
    total = sum(scores.values()) or 1.0
    return best_family, round(min(1.0, best_score / total + 0.35), 3)


def analyze_file(path: Path) -> GameSourceCard:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        analyzer = EnvironmentSourceAnalyzer(path, source)
        analyzer.visit(tree)
        return analyzer.build_card()
    except Exception as exc:
        return GameSourceCard(
            game_id=infer_game_id(path),
            path=str(path),
            parse_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def find_python_files(root: Path, game_ids: set[str] | None = None) -> list[Path]:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not game_ids:
        return files
    return [path for path in files if infer_game_id(path) in game_ids or path.stem in game_ids]


def default_root() -> Path:
    candidates = [
        os.getenv("ENVIRONMENTS_DIR"),
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
        "environment_files",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path("environment_files")


def print_text(cards: list[GameSourceCard]) -> None:
    if not cards:
        print("No Python environment files found.")
        return
    rows = []
    for card in cards:
        grid = ",".join(f"{w}x{h}" for w, h in card.grid_sizes) or "-"
        actions = ",".join(card.action_refs) or "-"
        hints = ",".join(card.source_hints) or "-"
        rows.append(
            [
                card.game_id,
                "ok" if card.parse_ok else "err",
                str(card.levels),
                grid,
                str(len(card.sprites)),
                actions,
                card.control_family,
                card.grid_bucket,
                card.task_family_prior,
                f"{card.confidence:.2f}",
                card.recommended_strategy,
                hints,
            ]
        )
    headers = [
        "game",
        "parse",
        "levels",
        "grid",
        "sprites",
        "actions",
        "control",
        "bucket",
        "prior",
        "conf",
        "strategy",
        "focused_hints",
    ]
    widths = [max(len(row[i]) for row in rows + [headers]) for i in range(len(headers))]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def write_json(cards: list[GameSourceCard], output: Path | None) -> None:
    payload = [asdict(card) for card in cards]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(text)
    else:
        output.write_text(text + "\n", encoding="utf-8")


def write_csv(cards: list[GameSourceCard], output: Path | None) -> None:
    fieldnames = [
        "game_id",
        "path",
        "parse_ok",
        "levels",
        "grid_sizes",
        "sprite_count",
        "action_refs",
        "uses_action6",
        "uses_action7",
        "uses_action_input",
        "control_family",
        "grid_bucket",
        "recommended_strategy",
        "source_hints",
        "broad_source_hints",
        "action_context_hints",
        "goal_context_hints",
        "task_family_prior",
        "confidence",
        "error",
    ]
    target = output.open("w", encoding="utf-8", newline="") if output is not None else sys.stdout
    close_target = output is not None
    try:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            writer.writerow(
                {
                    "game_id": card.game_id,
                    "path": card.path,
                    "parse_ok": card.parse_ok,
                    "levels": card.levels,
                    "grid_sizes": ";".join(f"{w}x{h}" for w, h in card.grid_sizes),
                    "sprite_count": len(card.sprites),
                    "action_refs": ";".join(card.action_refs),
                    "uses_action6": card.uses_action6,
                    "uses_action7": card.uses_action7,
                    "uses_action_input": card.uses_action_input,
                    "control_family": card.control_family,
                    "grid_bucket": card.grid_bucket,
                    "recommended_strategy": card.recommended_strategy,
                    "source_hints": ";".join(card.source_hints),
                    "broad_source_hints": ";".join(card.broad_source_hints),
                    "action_context_hints": ";".join(card.action_context_hints),
                    "goal_context_hints": ";".join(card.goal_context_hints),
                    "task_family_prior": card.task_family_prior,
                    "confidence": card.confidence,
                    "error": card.error or "",
                }
            )
    finally:
        if close_target:
            target.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze public ARC-AGI-3 environment Python sources.")
    parser.add_argument("--root", type=Path, default=default_root(), help="Root directory containing environment_files.")
    parser.add_argument("--game-id", action="append", default=[], help="Optional game id filter, e.g. bp35. Can repeat.")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for json/csv.")
    parser.add_argument("--limit", type=int, default=0, help="Analyze only the first N files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = find_python_files(args.root, set(args.game_id) if args.game_id else None)
    if args.limit > 0:
        files = files[: args.limit]
    cards = [analyze_file(path) for path in files]
    cards.sort(key=lambda card: (card.game_id, card.path))
    if args.format == "json":
        write_json(cards, args.output)
    elif args.format == "csv":
        write_csv(cards, args.output)
    else:
        print_text(cards)


if __name__ == "__main__":
    main()
