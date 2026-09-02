#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

state_root=${REVIEWER_VALIDATION_ROOT:-results/reviewer_validation}
summary_dir="$state_root/summary_current"
marker="$state_root/FINALIZED_PATH"
lock_dir="$state_root/.finalization.lock"

if [[ -s $marker ]]; then
    finalized_path=$(<"$marker")
    if [[ -d $finalized_path ]]; then
        echo "[finalize] already frozen at $finalized_path"
        exit 0
    fi
    echo "[finalize] marker points to a missing directory: $finalized_path" >&2
    exit 1
fi

if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "[finalize] another finalizer owns $lock_dir"
    exit 0
fi

staging=
cleanup() {
    [[ -z $staging ]] || rm -rf "$staging"
    rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT

stamp=$(date -u +%Y%m%dT%H%M%SZ)
destination="$state_root/final_10seed_$stamp"
staging=$(mktemp -d "$state_root/.finalizing.XXXXXX")
mkdir -p "$staging/summary" "$staging/publication"

# Both commands enforce the exact ten-paired-seed publication matrix. Running
# them again here prevents a stale publication_current directory from being
# mistaken for a completed result.
uv run python scripts/workshop/plot_continuous_control_results.py \
    --summary-dir "$summary_dir" \
    --output-dir "$staging/publication"
uv run python -m scripts.workshop.render_continuous_control_report \
    --summary-dir "$summary_dir" \
    --output-dir "$staging/publication"

cp "$summary_dir"/*.csv "$summary_dir"/*.json "$staging/summary/"
cp "$state_root/FIVE_NODE_EXECUTION_MANIFEST.md" "$staging/"
cp "$state_root/INTERIM_PUBLICATION_DECISIONS.md" "$staging/"
cp "$state_root/publication_gate.log" "$staging/"

find "$state_root/raw" -type f -name '*.json' -print0 \
    | sort -z \
    | while IFS= read -r -d '' path; do
        hash=$(sha256sum "$path" | cut -d ' ' -f 1)
        printf '%s  %s\n' "$hash" "$path"
    done >"$staging/RAW_ARTIFACT_SHA256SUMS.txt"

generated_at=$(date -u --iso-8601=seconds)
git_commit=$(git rev-parse HEAD)
raw_file_count=$(find "$state_root/raw" -type f -name '*.json' | wc -l)
raw_bytes=$(find "$state_root/raw" -type f -name '*.json' -printf '%s\n' \
    | awk '{total += $1} END {print total + 0}')
GENERATED_AT="$generated_at" GIT_COMMIT="$git_commit" \
RAW_FILE_COUNT="$raw_file_count" RAW_BYTES="$raw_bytes" \
    uv run python - <<'PY' >"$staging/FINALIZATION_METADATA.json"
import json
import os

print(
    json.dumps(
        {
            "generated_at_utc": os.environ["GENERATED_AT"],
            "git_commit": os.environ["GIT_COMMIT"],
            "minimum_paired_seeds": 10,
            "required_seed_ids": list(range(10)),
            "raw_json_file_count": int(os.environ["RAW_FILE_COUNT"]),
            "raw_json_bytes": int(os.environ["RAW_BYTES"]),
        },
        indent=2,
        sort_keys=True,
    )
)
PY

(
    cd "$staging"
    find . -type f ! -name SHA256SUMS.txt -print0 \
        | sort -z \
        | xargs -0 sha256sum >SHA256SUMS.txt
)

mv "$staging" "$destination"
staging=
marker_tmp="$marker.tmp"
printf '%s\n' "$destination" >"$marker_tmp"
mv "$marker_tmp" "$marker"
echo "[finalize] froze complete validation at $destination"
