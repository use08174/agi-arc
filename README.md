# ARC-AGI-3 Starter

Research-oriented starter scaffold for building ARC-AGI-3 agents around explicit exploration, state graphs, and reusable world models.

## Architecture

This scaffold is organized around a simple loop:

1. `envs/` adapts an environment into a stable internal interface.
2. `perception/` converts raw frames into normalized observations and state keys.
3. `memory/` stores transitions, loops, action outcomes, and reusable game knowledge.
4. `exploration/` proposes information-gathering actions early in a run.
5. `planning/` switches to goal-directed search once the agent has enough structure.
6. `agents/` orchestrates the full policy for one game.
7. `runner/` wires everything together for local experiments.

The design assumes ARC-AGI-3 success will come from:

- fast elimination of no-op or looping actions
- explicit state-graph construction
- level-to-level transfer inside a game
- small learned heuristics helping symbolic search, not replacing it

## Folder Layout

```text
src/arc_agi3/
  agents/        Agent orchestration
  core/          Shared types and config
  envs/          Environment adapter interfaces
  exploration/   Explore-first policies
  memory/        State graph and cross-level memory
  perception/    Observation normalization and hashing
  planning/      Goal-directed search
  runner/        Local entrypoints
environment_files/
  ms00/          Tiny local ARC-compatible environment for offline smoke tests
tests/           Unit and integration smoke tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install setuptools
python -m pip install --no-build-isolation -e .
python -m arc_agi3.runner.main --backend mock
```

To use the official ARC toolkit too:

```bash
python -m pip install arc-agi
python -m arc_agi3.runner.main --backend arcade --mode offline --game-id ms00
```

If you want online ARC environments, add `ARC_API_KEY` to `.env` and switch `--mode online` or `--mode competition`.

## Commands

List available local or online games:

```bash
python -m arc_agi3.runner.main --backend arcade --mode offline --list-games
```

Run the bundled offline demo environment:

```bash
python -m arc_agi3.runner.main --backend arcade --mode offline --game-id ms00
```

Run against the official API so scorecards and replays are available:

```bash
python -m arc_agi3.runner.main --backend arcade --mode online --game-id ls20
```

After a run, the runner prints:

- `scorecard_id`
- `score`
- `scorecard_url`
- whether online replays are available

Fetch a scorecard again later:

```bash
python -m arc_agi3.runner.scorecard --scorecard-id YOUR_CARD_ID --mode online
```

Run the mock environment:

```bash
python -m arc_agi3.runner.main --backend mock
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Real ARC Integration

The real adapter uses the official `arc-agi` package and follows the current toolkit interface:

- `Arcade.make(...)` for environment creation
- `env.action_space` for valid actions
- `env.step(action, data=...)` for action execution
- `OperationMode.OFFLINE | ONLINE | COMPETITION`

Internally, the adapter converts ARC frames into our simpler `Frame` dataclass using the final rendered frame as the current state representation.

## Score Checking

Official scorecards and replay pages are available for API-backed runs. ARC docs note that:

- API runs can be viewed online through a scorecard page
- local toolkit runs do not produce online replays
- the official benchmarking repo also stores scorecards on the ARC server

So the practical flow is:

1. Run with `--mode online` or `--mode competition`
2. Copy the printed `scorecard_id`
3. Open the printed `scorecard_url`
4. Re-fetch it later with `python -m arc_agi3.runner.scorecard --scorecard-id ...`

## Next Steps

1. Replace the coarse `StateHasher` with object-centric features and diff regions.
2. Teach the click candidate generator to mine changed pixels and object centroids.
3. Upgrade `SimplePlanner` to best-first search or MCTS over the explicit graph.
4. Add replay analysis and per-game experiment tracking.
5. Add learned value ranking for state-action frontier ordering.
