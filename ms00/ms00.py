from arcengine import ARCBaseGame, GameAction, Level, Sprite


class Ms00(ARCBaseGame):
    """Small local environment for offline adapter smoke tests.

    Reach the goal across three levels.
    ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right.
    """

    def __init__(self, seed: int = 0) -> None:
        levels = [self._make_level(goal_x=1, goal_y=2)]
        levels.append(self._make_level(goal_x=2, goal_y=2))
        levels.append(self._make_level(goal_x=2, goal_y=1))
        super().__init__(
            game_id="ms00",
            levels=levels,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def _make_level(self, goal_x: int, goal_y: int) -> Level:
        player = Sprite([[2]], name="player", x=1, y=1, tags=["player"])
        goal = Sprite([[3]], name="goal", x=goal_x, y=goal_y, tags=["goal"])
        return Level(sprites=[player, goal], grid_size=(4, 4))

    def step(self) -> None:
        player = self.current_level.get_sprites_by_name("player")[0]
        dx = 0
        dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
            dx = 1

        nx = max(0, min(3, player.x + dx))
        ny = max(0, min(3, player.y + dy))
        player.set_position(nx, ny)

        goal = self.current_level.get_sprites_by_name("goal")[0]
        if player.x == goal.x and player.y == goal.y:
            self.next_level()

        self.complete_action()
