#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v29"
model_dir="$project_root/vision-artifacts/model"
deployed_checkpoint="$model_dir/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
reference_checkpoint="$artifact_root/deployed-active-v12.ftsv"
reference_head="$artifact_root/frozen-active-v13-head.json"

[[ $(cat "$artifact_root/browser-selection.txt") == selected_candidate=high-resolution ]] \
  || { echo 'active-v29 high-resolution candidate was not selected' >&2; exit 2; }
for replica in 1 2 3 4; do
  [[ -s "$artifact_root/high-resolution/replica-$replica/set-vision.ftsv" ]] \
    || { echo "missing high-resolution replica $replica" >&2; exit 2; }
done
expected_primary=$(shasum -a 256 "$artifact_root/high-resolution/replica-1/set-vision.ftsv" | awk '{ print $1 }')
actual_primary=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
[[ "$actual_primary" == "$expected_primary" ]] || { echo 'selected primary checkpoint is not installed' >&2; exit 2; }

audit_dir="$artifact_root/audit"
[[ ! -e "$audit_dir" ]] || { echo 'active-v29 audit already exists' >&2; exit 2; }
mkdir -p "$audit_dir"
pids=()
pending=0
for start in 0 250 500 750 1000 1250 1500 1750; do
  node "$project_root/tools/test-vision-extension.mjs" \
    --stream --audit --boards 250 --start "$start" --seed 2026090999 --report-every 250 \
    >"$audit_dir/shard-$start.log" 2>&1 &
  pids+=("$!")
  pending=$((pending + 1))
  if ((pending == 4)); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
    pending=0
  fi
done

failures=0
has_correct=0
no_correct=0
accepted=0
accepted_correct=0
for start in 0 250 500 750 1000 1250 1500 1750; do
  summary=$(jq -c 'select(.event == "stream_summary")' "$audit_dir/shard-$start.log" | tail -n 1)
  shard_failures=$(jq -r '.failures' <<<"$summary")
  has_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
  no_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
  coverage=$(jq -r '.guardedCoverage' <<<"$summary")
  guarded_accuracy=$(jq -r '.guardedAccuracy // 0' <<<"$summary")
  shard_has_correct=$(awk -v value="$has_accuracy" 'BEGIN { printf "%.0f", value * 125 }')
  shard_no_correct=$(awk -v value="$no_accuracy" 'BEGIN { printf "%.0f", value * 125 }')
  shard_accepted=$(awk -v value="$coverage" 'BEGIN { printf "%.0f", value * 250 }')
  shard_accepted_correct=$(awk -v value="$guarded_accuracy" -v count="$shard_accepted" \
    'BEGIN { printf "%.0f", value * count }')
  failures=$((failures + shard_failures))
  has_correct=$((has_correct + shard_has_correct))
  no_correct=$((no_correct + shard_no_correct))
  accepted=$((accepted + shard_accepted))
  accepted_correct=$((accepted_correct + shard_accepted_correct))
done

has_failures=$((1000 - has_correct))
no_failures=$((1000 - no_correct))
raw_accuracy=$(awk -v value="$failures" 'BEGIN { print (2000 - value) / 2000 }')
has_accuracy=$(awk -v value="$has_correct" 'BEGIN { print value / 1000 }')
no_accuracy=$(awk -v value="$no_correct" 'BEGIN { print value / 1000 }')
guarded_accuracy=$(awk -v value="$accepted_correct" -v count="$accepted" \
  'BEGIN { if (count == 0) print 0; else print value / count }')
guarded_coverage=$(awk -v value="$accepted" 'BEGIN { print value / 2000 }')
passed=no
if ((failures <= 40 && has_failures <= 25 && no_failures <= 25)); then passed=yes; fi

printf 'seed\tboards\traw_accuracy\tfailures\thas_set_accuracy\thas_set_failures\tno_set_accuracy\tno_set_failures\tguarded_accuracy\tguarded_coverage\tpassed\n' \
  >"$artifact_root/audit-summary.tsv"
printf '2026090999\t2000\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$raw_accuracy" "$failures" "$has_accuracy" "$has_failures" "$no_accuracy" "$no_failures" \
  "$guarded_accuracy" "$guarded_coverage" "$passed" >>"$artifact_root/audit-summary.tsv"

if [[ "$passed" == no ]]; then
  cp "$reference_checkpoint" "$deployed_checkpoint"
  cp "$reference_head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
fi
column -t -s $'\t' "$artifact_root/audit-summary.tsv"
printf 'audit_passed=%s\n' "$passed" | tee "$artifact_root/audit-selection.txt"
