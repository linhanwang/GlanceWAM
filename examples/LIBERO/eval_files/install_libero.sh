#!/bin/bash
# Install the LIBERO evaluation environment with uv.
#
# LIBERO is managed as a standalone uv project (Python 3.8.13, torch cu113).
# `uv sync` creates LIBERO/.venv, installs the pinned deps + the editable
# `libero` package, and pulls mujoco 3.2.3 transitively via robosuite 1.4.0.
# The rollout client (examples/LIBERO/eval_files/eval_libero.py) runs in this
# venv, so we add the few extras it needs that are outside LIBERO's lockfile.
set -e

# Resolve repo root from this script's location; LIBERO lives next to glancewam.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLANCEWAM_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LIBERO_DIR="$(dirname "${GLANCEWAM_DIR}")/LIBERO"

echo "=== Step 1: Ensure uv is installed ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "=== Step 2: Clone LIBERO (uv-managed fork) ==="
if [ ! -d "$LIBERO_DIR" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
else
    echo "LIBERO already present at $LIBERO_DIR"
fi
cd "$LIBERO_DIR"

echo "=== Step 3: Sync the environment (creates .venv, installs libero editable) ==="
# Pins Python 3.8.13 + torch/vision/audio 1.11.0+cu113 (PyTorch cu113 index).
# Also installs robosuite 1.4.0 -> mujoco 3.2.3 and imageio transitively.
uv sync

echo "=== Step 4: Install eval-client extras (not in LIBERO's lockfile) ==="
# Needed by eval_libero.py / model2libero_interface.py to talk to the policy
# server. `rich` is pulled in transitively when the client imports
# glancewam.model.tools (overwatch logging) — that import also requires the venv
# be Python >=3.10, which is why LIBERO's pyproject is pinned to 3.10.
uv pip install tyro websockets msgpack rich

echo "=== Step 5: Verify installation ==="
uv run python -c "from libero.libero import benchmark; print('LIBERO OK:', benchmark)"
uv run python -c "import mujoco; print('MuJoCo OK:', mujoco.__version__)"
uv run python -c "import tyro, websockets, msgpack, imageio; print('eval deps OK')"

echo "=== ALL DONE ==="
echo "LIBERO venv python: ${LIBERO_DIR}/.venv/bin/python"
echo "Point LIBERO_Python in eval_libero.sh at that interpreter."
