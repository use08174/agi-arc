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

## Local LLM Hook On Kaggle

Recommended first model choices for this repo:

1. `Qwen2.5-Coder-1.5B-Instruct`
2. `Qwen2.5-Coder-3B-Instruct`
3. `Qwen2.5-7B-Instruct` only if you can tolerate slower calls

These are best used for:

- ranking a shortlist of candidate actions
- proposing short rule hypotheses

not for directly controlling every step.

Example run with a local model mounted as Kaggle input:

```python
!python /kaggle/input/arc-agi3-starter/scripts/kaggle_local_llm_ls20.py \
    --mode online \
    --game-id ls20 \
    --model-short-name qwen-coder-1.5b \
    --llm-start-step 8 \
    --llm-step-interval 8 \
    --llm-max-calls 12
```

The current implementation already throttles calls by:

- waiting until `start_step`
- only calling every `step_interval` steps
- capping calls per episode
- caching decisions by state/action signature

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
