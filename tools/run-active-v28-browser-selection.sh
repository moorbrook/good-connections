#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v28"
deployed_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
reference_checkpoint="$artifact_root/deployed-reference-set-vision.ftsv"
reference_head="$artifact_root/frozen-board-head.json"
candidate_checkpoint="$artifact_root/w075/set-vision.ftsv"
results="$artifact_root/browser-selection.tsv"

expected_checkpoint_hash=ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec
expected_head_hash=993931b8fa4386d4e66ab71619b16d6c113672beb9387312addb8aed53ebde90
actual_checkpoint_hash=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
actual_head_hash=$(shasum -a 256 "$deployed_head" | awk '{ print $1 }')
[[ "$actual_checkpoint_hash" == "$expected_checkpoint_hash" ]] || { echo "unexpected deployed checkpoint" >&2; exit 2; }
[[ "$actual_head_hash" == "$expected_head_hash" ]] || { echo "unexpected deployed head" >&2; exit 2; }

cp "$deployed_checkpoint" "$reference_checkpoint"
cp "$deployed_head" "$reference_head"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' >"$results"

candidates=(deployed-reference w075)
checkpoints=("$reference_checkpoint" "$candidate_checkpoint")
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  checkpoint=${checkpoints[$index]}
  cp "$checkpoint" "$deployed_checkpoint"
  cp "$reference_head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  shard_pids=()
  for start in 0 256; do
    log="$artifact_root/$candidate/browser-development-$start.log"
    mkdir -p "$(dirname "$log")"
    node "$project_root/tools/test-vision-extension.mjs" \
      --stream --audit --boards 256 --start "$start" --seed 2026090701 --report-every 256 >"$log" 2>&1 &
    shard_pids+=("$!")
  done
  for pid in "${shard_pids[@]}"; do wait "$pid"; done

  board_failures=0
  has_set_correct=0
  no_set_correct=0
  for start in 0 256; do
    log="$artifact_root/$candidate/browser-development-$start.log"
    summary=$(jq -c 'select(.event == "stream_summary")' "$log" | tail -n 1)
    shard_failures=$(jq -r '.failures' <<<"$summary")
    shard_has_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
    shard_no_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
    shard_has_correct=$(awk -v accuracy="$shard_has_accuracy" 'BEGIN { printf "%.0f", accuracy * 128 }')
    shard_no_correct=$(awk -v accuracy="$shard_no_accuracy" 'BEGIN { printf "%.0f", accuracy * 128 }')
    board_failures=$((board_failures + shard_failures))
    has_set_correct=$((has_set_correct + shard_has_correct))
    no_set_correct=$((no_set_correct + shard_no_correct))
  done
  board_accuracy=$(awk -v failures="$board_failures" 'BEGIN { print (512 - failures) / 512 }')
  board_set_accuracy=$(awk -v correct="$has_set_correct" 'BEGIN { print correct / 256 }')
  board_no_set_accuracy=$(awk -v correct="$no_set_correct" 'BEGIN { print correct / 256 }')
  worst_conditional=$(awk -v has_acc="$board_set_accuracy" -v no_acc="$board_no_set_accuracy" \
    'BEGIN { if (has_acc < no_acc) print has_acc; else print no_acc }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$worst_conditional" >>"$results"
done

winner=$(awk -F '\t' '
  NR == 2 { reference_failures = $3; reference_has = $4; reference_no = $5 }
  NR == 3 {
    tolerance = 1 / 256
    if ($3 < reference_failures && $4 + tolerance >= reference_has && $5 + tolerance >= reference_no) print $1
    else print "deployed-reference"
  }
' "$results")

if [[ "$winner" == w075 ]]; then cp "$candidate_checkpoint" "$deployed_checkpoint"
else cp "$reference_checkpoint" "$deployed_checkpoint"
fi
cp "$reference_head" "$deployed_head"
"$project_root/tools/build-vision-extension.sh" >/dev/null
column -t -s $'\t' "$results"
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
