#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PARENT_PID CHILD_PID" >&2
    exit 2
fi

parent_pid=$1
child_pid=$2

while [[ -r "/proc/${child_pid}/stat" ]]; do
    state=$(cut -d ' ' -f 3 "/proc/${child_pid}/stat")
    [[ "$state" != "Z" ]] || break
    sleep 60
done

# The queue parent is stopped while its current child finishes. Retiring it
# here prevents it from launching a seed reassigned to another node.
kill -KILL "$parent_pid" 2>/dev/null || true
