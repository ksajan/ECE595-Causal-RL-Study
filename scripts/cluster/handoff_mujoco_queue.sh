#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 PARENT_PID CHILD_PID ENV_ID VARIANT START_SEED END_SEED" >&2
    exit 2
fi

parent_pid=$1
child_pid=$2
env_id=$3
variant=$4
start_seed=$5
end_seed=$6

while [[ -r "/proc/${child_pid}/stat" ]]; do
    state=$(cut -d ' ' -f 3 "/proc/${child_pid}/stat")
    [[ "$state" != "Z" ]] || break
    sleep 60
done

# The old parent is deliberately stopped so it cannot launch an overlapping
# seed. Terminate it only after its current child has exited.
kill -KILL "$parent_pid" 2>/dev/null || true

cd "$(dirname "$0")/../.."
exec bash scripts/workshop/queue_mujoco_oracle_seeds.sh \
    "$env_id" "$variant" "$start_seed" "$end_seed"
