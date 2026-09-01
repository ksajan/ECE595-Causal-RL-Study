#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 DATASET VARIANT FIRST_SEED LAST_SEED" >&2
    exit 2
fi

dataset=$1
variant=$2
first_seed=$3
last_seed=$4

python_bin=${PYTHON_BIN:-.venv/bin/python}
hdf5_path=${HDF5_PATH:-data/d4rl_hdf5/$dataset.hdf5}
augmentation_cache_dir=${AUGMENTATION_CACHE_DIR:-data/d4rl_simulator_cf_v2}
augmentation_seed=${AUGMENTATION_SEED:-0}
output_root=${OUTPUT_ROOT:-results/d4rl_cql_full}
log_root=${LOG_ROOT:-logs/d4rl_cql_full}
steps=${STEPS:-500000}
steps_per_epoch=${STEPS_PER_EPOCH:-10000}
eval_episodes=${EVAL_EPISODES:-50}
eval_seed_base=${EVAL_SEED_BASE:-940000}
intervention_scale=${INTERVENTION_SCALE:-0.10}
cf_fraction=${CF_FRACTION:-0.50}
wait_for_evaluation=${WAIT_FOR_EVALUATION:-}
wait_timeout_seconds=${WAIT_TIMEOUT_SECONDS:-21600}
min_normalized_score=${MIN_NORMALIZED_SCORE:-}

mkdir -p "$log_root"

if [[ -n "$wait_for_evaluation" ]]; then
    waited=0
    until [[ -f "$wait_for_evaluation" ]]; do
        if (( waited >= wait_timeout_seconds )); then
            echo "[queue] timed out waiting for: $wait_for_evaluation" >&2
            exit 1
        fi
        sleep 60
        waited=$((waited + 60))
    done
    if [[ -n "$min_normalized_score" ]]; then
        "$python_bin" - "$wait_for_evaluation" "$min_normalized_score" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
threshold = float(sys.argv[2])
score = float(json.loads(path.read_text(encoding="utf-8"))["normalized_mean"])
print(f"[queue] baseline gate: normalized={score:.3f}, required={threshold:.3f}")
if score < threshold:
    raise SystemExit(1)
PY
    fi
fi

for seed in $(seq "$first_seed" "$last_seed"); do
    run_dir="$output_root/$dataset/$variant/seed_$seed"
    if [[ -f "$run_dir/evaluation.json" ]]; then
        echo "[queue] already complete: $dataset $variant seed=$seed"
        continue
    fi
    if [[ -e "$run_dir" ]]; then
        echo "[queue] refusing incomplete existing run: $run_dir" >&2
        exit 1
    fi

    echo "[queue] starting: $dataset $variant seed=$seed"
    "$python_bin" scripts/workshop/run_d4rl_simulator_cf_cql.py \
        --mode train \
        --dataset "$dataset" \
        --variant "$variant" \
        --seed "$seed" \
        --augmentation-seed "$augmentation_seed" \
        --hdf5-path "$hdf5_path" \
        --augmentation-cache-dir "$augmentation_cache_dir" \
        --intervention-scale "$intervention_scale" \
        --cf-fraction "$cf_fraction" \
        --steps "$steps" \
        --steps-per-epoch "$steps_per_epoch" \
        --eval-episodes "$eval_episodes" \
        --eval-seed-base "$eval_seed_base" \
        --output-root "$output_root" \
        --quiet \
        >"$log_root/${dataset}_${variant}_seed${seed}.log" 2>&1
done
