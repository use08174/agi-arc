# Kaggle Guide

This repo can be used on Kaggle in two different modes:

1. Interactive research notebook with `Internet ON`
2. Competition-style or reproducibility notebook with `Internet OFF`

These are not the same.

## What Works Where

`Internet ON`

- `pip install arc-agi`
- ARC API access via `ARC_API_KEY`
- official online `scorecard` pages
- online replays for API runs

`Internet OFF`

- local repo code from `src/`
- bundled offline demo environment `ms00`
- local tests
- no ARC API scorecard/replay fetching
- no `pip install arc-agi` from PyPI unless you upload wheels as Kaggle input

## Recommended Way To Move The Code

The most reliable Kaggle workflow is:

1. Upload this repo as a Kaggle Dataset, or add it as a Notebook Input.
2. In the notebook, append `src/` to `sys.path` instead of relying on editable installs.
3. Use the provided wrapper scripts in `scripts/`.

Example if the repo is mounted at `/kaggle/input/arc-agi3-starter`:

```python
import os
import sys
from pathlib import Path

REPO = Path("/kaggle/input/arc-agi3-starter")
sys.path.insert(0, str(REPO / "src"))

os.environ["MPLCONFIGDIR"] = "/kaggle/working/.arc_agi3_cache/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/kaggle/working/.arc_agi3_cache/xdg-cache"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
```

## Kaggle Settings

For an online public-game run like `ls20`:

1. Turn `Internet` to `ON`
2. Add a Kaggle secret named `ARC_API_KEY`, or set it manually in a cell
3. Optionally install `arc-agi`

Example:

```python
import os
os.environ["ARC_API_KEY"] = "YOUR_KEY"
```

## Fastest Online Test On Kaggle

If internet is on and the repo is available locally:

```python
!python /kaggle/input/arc-agi3-starter/scripts/kaggle_online_ls20.py \
    --install-mode online \
    --mode online \
    --game-id ls20
```

This prints:

- `scorecard_id`
- `score`
- `scorecard_url`
- replay availability

## Re-check A Scorecard Later

```python
!python /kaggle/input/arc-agi3-starter/scripts/kaggle_scorecard.py \
    --scorecard-id YOUR_CARD_ID \
    --mode online
```

## Internet-Off Notebook

If internet is off, use the local demo game:

```python
!python /kaggle/input/arc-agi3-starter/scripts/kaggle_online_ls20.py \
    --install-mode skip \
    --mode offline \
    --game-id ms00
```

This verifies the architecture only. It does not create official ARC scorecards or replays.

## Offline Evaluation Notebook

For the notebook `arc3-agent-evaluation-and-recording-viewer.ipynb`, use the official-interface
adapter in this repo instead of the random baseline agent:

```python
import os
from pathlib import Path

AGI_ARC_REPO = Path("/kaggle/working/agi-arc")
os.environ["AGI_ARC_REPO"] = str(AGI_ARC_REPO)

result = run_arc(
    agent_src=AGI_ARC_REPO / "scripts" / "kaggle_offline_eval_agent.py",
    agent_class_name="MyAgent",
    agent_cli_name="myagent",
    run_game="all",
    description="agi-arc-offline-v1",
)
```

This keeps the notebook's official `main.py` execution path, while the adapter translates
frame-by-frame `choose_action(...)` calls into this repo's graph-search runtime.

Optional environment variables for the adapter:

```python
os.environ["ARC_AGI3_MAX_STEPS"] = "128"
os.environ["ARC_AGI3_EXPLORE_STEPS"] = "24"
```

Use `--game-id all` to evaluate every public offline environment in `environment_files`.

## If You Need `arc-agi` In Internet-Off Mode

You need to upload wheel files as a Kaggle Dataset first, then install from that dataset.

Example:

```python
!python /kaggle/input/arc-agi3-starter/scripts/kaggle_online_ls20.py \
    --install-mode wheel-dir \
    --wheel-dir /kaggle/input/your-wheel-dataset \
    --mode offline \
    --game-id ms00
```

For official online scorecards, internet still needs to be on because the ARC API must be reachable.

## Important Limitation

Official ARC scorecards and replays only exist for API-backed runs. If Kaggle internet is off, the ARC API cannot be reached, so only local execution is possible.
