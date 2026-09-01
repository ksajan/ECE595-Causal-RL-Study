#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 ENV_ID VARIANT FIRST_SEED LAST_SEED" >&2
    exit 2
fi

env_id=$1
variant=$2
first_seed=$3
last_seed=$4

python_bin=${PYTHON_BIN:-.venv/bin/python}
output_root=${OUTPUT_ROOT:-results/mujoco_oracle_full}
log_root=${LOG_ROOT:-logs/mujoco_full}
job_id=${JOB_ID:-full1m-v1}
total_timesteps=${TOTAL_TIMESTEPS:-1000000}
eval_episodes=${EVAL_EPISODES:-100}
eval_seed_base=${EVAL_SEED_BASE:-930000}
learning_starts=${LEARNING_STARTS:-10000}
buffer_size=${BUFFER_SIZE:-2000000}
batch_size=${BATCH_SIZE:-256}
wait_for_path=${WAIT_FOR_PATH:-}
wait_timeout_seconds=${WAIT_TIMEOUT_SECONDS:-21600}

mkdir -p "$log_root"

if [[ -n "$wait_for_path" ]]; then
    waited=0
    until [[ -f "$wait_for_path" ]]; do
        if (( waited >= wait_timeout_seconds )); then
            echo "[queue] timed out waiting for: $wait_for_path" >&2
            exit 1
        fi
        sleep 60
        waited=$((waited + 60))
    done
fi

for seed in $(seq "$first_seed" "$last_seed"); do
    run_dir="$output_root/$env_id/$variant/seed_$seed/$job_id"
    if [[ -f "$run_dir/eval.json" ]]; then
        echo "[queue] already complete: $env_id $variant seed=$seed"
        continue
    fi
    if [[ -e "$run_dir" ]]; then
        echo "[queue] refusing incomplete existing run: $run_dir" >&2
        exit 1
    fi

    echo "[queue] starting: $env_id $variant seed=$seed"
    "$python_bin" scripts/workshop/run_mujoco_oracle_cf_sac.py \
        --env-id "$env_id" \
        --seed "$seed" \
        --variant "$variant" \
        --total-timesteps "$total_timesteps" \
        --eval-episodes "$eval_episodes" \
        --eval-seed-base "$eval_seed_base" \
        --learning-starts "$learning_starts" \
        --buffer-size "$buffer_size" \
        --batch-size "$batch_size" \
        --output-root "$output_root" \
        --job-id "$job_id" \
        >"$log_root/${env_id}_${variant}_seed${seed}.log" 2>&1
done
