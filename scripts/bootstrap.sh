#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install setuptools
python -m pip install --no-build-isolation -e .

if [[ "${1:-}" == "--with-arc" ]]; then
  python -m pip install arc-agi
fi

echo "Bootstrap complete."
