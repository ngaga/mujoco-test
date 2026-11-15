#!/usr/bin/env bash
# Install an ISOLATED Python 3.12 VLA environment; do not touch EarlyBots .venv.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required. On macOS: brew install python@3.12" >&2
  exit 1
fi

VENV=".venv-vla"
if [ ! -x "$VENV/bin/python" ]; then
  python3.12 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  "git+https://github.com/ahmedsohail2003/workcell-partA-imitation.git@5822ac1dd508c92183fd8ecc99f91d6ec386549b"
"$VENV/bin/python" -m pip install 'lerobot[smolvla]==0.6.0'

echo
echo "SO-101 SmolVLA dependencies installed in $VENV."
echo "From the repository root:"
echo "  source .venv-vla/bin/activate"
echo "  mjpython vla/run_so101.py --check-env"
echo "  mjpython vla/run_so101.py --episodes 1 --device mps --gif vla_runs/first_vla.gif"
