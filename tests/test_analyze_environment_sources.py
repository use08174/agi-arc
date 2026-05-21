from scripts.analyze_environment_sources import analyze_file


def test_analyze_environment_source_extracts_static_priors(tmp_path):
    game_dir = tmp_path / "bp35" / "0a0ad940"
    game_dir.mkdir(parents=True)
    source = game_dir / "bp35.py"
    source.write_text(
        """
from arcengine import ActionInput, GameAction, Level, Sprite

sprites = {
    "sprite-1": Sprite(pixels=[[9]], name="sprite-1", visible=True, collidable=True),
}
levels = [
    Level(sprites=[sprites["sprite-1"].clone().set_position(4, 3)], grid_size=(8, 8)),
    Level(sprites=[sprites["sprite-1"].clone().set_position(4, 3)], grid_size=(8, 8)),
]

def step(action: GameAction, action_input: ActionInput):
    if action == GameAction.ACTION6:
        return action_input.x, action_input.y
    if action == GameAction.ACTION1:
        return "move"
""",
        encoding="utf-8",
    )

    card = analyze_file(source)

    assert card.parse_ok is True
    assert card.game_id == "bp35"
    assert card.levels == 2
    assert card.grid_sizes == [[8, 8]]
    assert len(card.sprites) == 1
    assert card.sprites[0].colors == [9]
    assert card.uses_action6 is True
    assert card.uses_action_input is True
    assert "navigation" in card.source_hints
    assert card.task_family_prior in {"painting", "selection", "navigation"}
