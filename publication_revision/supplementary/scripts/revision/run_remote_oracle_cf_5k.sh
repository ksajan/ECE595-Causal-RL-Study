#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/../.."
mkdir -p logs results/revision/cartpole_oracle_cf_5k

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
    PYTHON_BIN=python
else
    PYTHON_BIN=python3
fi

if ! "$PYTHON_BIN" -c 'import torch, numpy' >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --clear .venv
        # shellcheck disable=SC1091
        . .venv/bin/activate
        PYTHON_BIN=python
        uv pip install numpy torch
    else
        echo "Missing torch/numpy and uv is unavailable." >&2
        exit 2
    fi
fi

for seed in "$@"; do
    "$PYTHON_BIN" scripts/revision/cartpole_oracle_cf.py \
        --seed "$seed" \
        --mode real \
        --train-steps 5000 \
        --eval-episodes 100 \
        --output-dir "results/revision/cartpole_oracle_cf_5k/real_seed_${seed}"
    "$PYTHON_BIN" scripts/revision/cartpole_oracle_cf.py \
        --seed "$seed" \
        --mode oracle_cf \
        --cf-frac 0.5 \
        --train-steps 5000 \
        --eval-episodes 100 \
        --output-dir "results/revision/cartpole_oracle_cf_5k/cf_seed_${seed}"
done
