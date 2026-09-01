#!/usr/bin/env bash
set -euo pipefail

interval_seconds=600
once=false

while (($#)); do
    case "$1" in
        --interval)
            interval_seconds=$2
            shift 2
            ;;
        --once)
            once=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if ! [[ $interval_seconds =~ ^[1-9][0-9]*$ ]]; then
    echo "--interval must be a positive integer" >&2
    exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

read -r -a nodes <<< "${REVIEWER_NODES:-1 2 3}"
remote_user=${REVIEWER_REMOTE_USER:-tasiuser}
remote_domain=${REVIEWER_REMOTE_DOMAIN:-boilerad.purdue.edu}
local_root=results/reviewer_validation
mkdir -p "$local_root/raw/mujoco" "$local_root/raw/d4rl" \
    "$local_root/summary_current"

poll_once() {
    printf '[%s] reviewer-validation poll\n' "$(date --iso-8601=seconds)"
    for node in "${nodes[@]}"; do
        host="x-indy-tasigpu${node}.${remote_domain}"
        target="${remote_user}@${host}"
        status=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$target" '
            cd ~/ece595-revision
            printf "SAC=%s " "$(find results/mujoco_oracle_full -name eval.json 2>/dev/null | wc -l)"
            printf "D4RL=%s " "$(find results/d4rl_cql_full -name evaluation.json 2>/dev/null | wc -l)"
            printf "active=%s " "$(pgrep -af "run_(mujoco_oracle_cf_sac|d4rl_simulator_cf_cql)" | grep -v queue | wc -l)"
            nvidia-smi --query-gpu=utilization.gpu,memory.used \
                --format=csv,noheader,nounits | awk -F, "{printf \"gpu=%s%% vram=%sMiB\", \$1, \$2}"
        ')
        printf 'node%s %s\n' "$node" "$status"

        rsync -a --prune-empty-dirs --include='*/' --include='*.json' \
            --exclude='*' "$target:~/ece595-revision/results/mujoco_oracle_full/" \
            "$local_root/raw/mujoco/"
        rsync -a --prune-empty-dirs --include='*/' --include='*.json' \
            --exclude='*' "$target:~/ece595-revision/results/d4rl_cql_full/" \
            "$local_root/raw/d4rl/"
    done

    uv run python scripts/workshop/summarize_continuous_control_results.py \
        --mujoco-root "$local_root/raw/mujoco" \
        --d4rl-root "$local_root/raw/d4rl" \
        --output-dir "$local_root/summary_current"
}

while true; do
    poll_once
    if $once; then
        break
    fi
    sleep "$interval_seconds"
done
