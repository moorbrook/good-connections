#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v28"
deployed_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
reference_checkpoint="$artifact_root/deployed-reference-set-vision.ftsv"
reference_head="$artifact_root/frozen-board-head.json"

expected_candidate_hash=0dad6fce0bf195a4ff752d5b9a36add8452aaeca95d03a6478c585736e41ed96
expected_head_hash=993931b8fa4386d4e66ab71619b16d6c113672beb9387312addb8aed53ebde90
actual_checkpoint_hash=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
actual_head_hash=$(shasum -a 256 "$deployed_head" | awk '{ print $1 }')
[[ "$actual_checkpoint_hash" == "$expected_candidate_hash" ]] || { echo "v28 candidate is not deployed" >&2; exit 2; }
[[ "$actual_head_hash" == "$expected_head_hash" ]] || { echo "active-v13 head is not deployed" >&2; exit 2; }

pids=()
for start in 0 250 500 750; do
  log="$artifact_root/audit-$start.log"
  node "$project_root/tools/test-vision-extension.mjs" \
    --stream --audit --boards 250 --start "$start" --seed 2026090799 --report-every 250 >"$log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

failures=0
has_correct=0
no_correct=0
guarded_accepted=0
guarded_correct=0
for start in 0 250 500 750; do
  summary=$(jq -c 'select(.event == "stream_summary")' "$artifact_root/audit-$start.log" | tail -n 1)
  shard_failures=$(jq -r '.failures' <<<"$summary")
  has_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
  no_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
  coverage=$(jq -r '.guardedCoverage' <<<"$summary")
  guarded_accuracy=$(jq -r '.guardedAccuracy // 0' <<<"$summary")
  shard_has_correct=$(awk -v accuracy="$has_accuracy" 'BEGIN { printf "%.0f", accuracy * 125 }')
  shard_no_correct=$(awk -v accuracy="$no_accuracy" 'BEGIN { printf "%.0f", accuracy * 125 }')
  shard_accepted=$(awk -v coverage="$coverage" 'BEGIN { printf "%.0f", coverage * 250 }')
  shard_guarded_correct=$(awk -v accuracy="$guarded_accuracy" -v accepted="$shard_accepted" \
    'BEGIN { printf "%.0f", accuracy * accepted }')
  failures=$((failures + shard_failures))
  has_correct=$((has_correct + shard_has_correct))
  no_correct=$((no_correct + shard_no_correct))
  guarded_accepted=$((guarded_accepted + shard_accepted))
  guarded_correct=$((guarded_correct + shard_guarded_correct))
done

has_failures=$((500 - has_correct))
no_failures=$((500 - no_correct))
raw_accuracy=$(awk -v failures="$failures" 'BEGIN { print (1000 - failures) / 1000 }')
has_accuracy=$(awk -v correct="$has_correct" 'BEGIN { print correct / 500 }')
no_accuracy=$(awk -v correct="$no_correct" 'BEGIN { print correct / 500 }')
guarded_coverage=$(awk -v accepted="$guarded_accepted" 'BEGIN { print accepted / 1000 }')
guarded_accuracy=$(awk -v correct="$guarded_correct" -v accepted="$guarded_accepted" \
  'BEGIN { if (accepted == 0) print 0; else print correct / accepted }')

passed=no
if [[ "$failures" -le 20 && "$has_failures" -le 12 && "$no_failures" -le 12 ]]; then
  passed=yes
else
  cp "$reference_checkpoint" "$deployed_checkpoint"
  cp "$reference_head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
fi

printf 'seed\tboards\traw_accuracy\tfailures\thas_set_accuracy\thas_set_failures\tno_set_accuracy\tno_set_failures\tguarded_accuracy\tguarded_coverage\tpassed\n' \
  >"$artifact_root/audit-summary.tsv"
printf '2026090799\t1000\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$raw_accuracy" "$failures" "$has_accuracy" "$has_failures" "$no_accuracy" "$no_failures" \
  "$guarded_accuracy" "$guarded_coverage" "$passed" >>"$artifact_root/audit-summary.tsv"
column -t -s $'\t' "$artifact_root/audit-summary.tsv"
printf 'audit_passed=%s\n' "$passed" | tee "$artifact_root/audit-selection.txt"
