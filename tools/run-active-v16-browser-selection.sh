#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v16"
deployed_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
results="$artifact_root/browser-selection.tsv"
raw_head="$artifact_root/raw-board-head.json"
original_checkpoint="$artifact_root/preselection-deployed-set-vision.ftsv"
original_head="$artifact_root/preselection-deployed-board-head.json"
control="$project_root/vision-artifacts/active-v10/rate-0/set-vision.ftsv"

expected_control_hash=dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e
actual_control_hash=$(shasum -a 256 "$control" | awk '{ print $1 }')
[[ "$actual_control_hash" == "$expected_control_hash" ]] || {
  echo "control checkpoint hash mismatch: $actual_control_hash" >&2
  exit 2
}

if [[ ! -s "$original_checkpoint" ]]; then
  cp "$deployed_checkpoint" "$original_checkpoint"
  cp "$deployed_head" "$original_head"
fi
printf '{\n  "type": "score-threshold",\n  "threshold": 0\n}\n' > "$raw_head"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' > "$results"

candidates=(control shape-m2 shape-m4)
checkpoints=(
  "$control"
  "$artifact_root/shape-m2/set-vision.ftsv"
  "$artifact_root/shape-m4/set-vision.ftsv"
)
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  checkpoint=${checkpoints[$index]}
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
  cp "$checkpoint" "$deployed_checkpoint"
  cp "$raw_head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  browser_log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$browser_log")"
  completed_boards=$(jq -r 'select(.event == "stream_summary") | .boards' "$browser_log" 2>/dev/null | tail -n 1 || true)
  if [[ "$completed_boards" != 512 ]]; then
    node "$project_root/tools/test-vision-extension.mjs" \
      --stream --audit --boards 512 --seed 2026082601 --report-every 512 > "$browser_log" 2>&1
  fi
  summary=$(jq -c 'select(.event == "stream_summary")' "$browser_log" | tail -n 1)
  board_accuracy=$(jq -r '.rawAccuracy' <<< "$summary")
  board_failures=$(jq -r '.failures' <<< "$summary")
  board_set_accuracy=$(jq -r '.hasSetAccuracy' <<< "$summary")
  board_no_set_accuracy=$(jq -r '.noSetAccuracy' <<< "$summary")
  worst_conditional=$(awk -v has_acc="$board_set_accuracy" -v no_acc="$board_no_set_accuracy" \
    'BEGIN { if (has_acc < no_acc) print has_acc; else print no_acc }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$worst_conditional" >> "$results"
done

winner=$(awk -F '\t' '
  NR == 2 { control_has = $4; control_no = $5 }
  NR > 1 {
    tolerance = 1 / 256
    eligible = ($1 == "control") || ($4 + tolerance >= control_has && $5 + tolerance >= control_no)
    multiplier = ($1 == "control") ? 1 : (($1 == "shape-m2") ? 2 : 4)
    if (eligible && (!selected || $3 < failures || ($3 == failures && ($6 > worst || ($6 == worst && multiplier < best_multiplier))))) {
      selected = 1
      candidate = $1
      failures = $3
      worst = $6
      best_multiplier = multiplier
    }
  }
  END { print candidate }
' "$results")

case "$winner" in
  control)
    cp "$original_checkpoint" "$deployed_checkpoint"
    cp "$original_head" "$deployed_head"
    ;;
  shape-m2)
    cp "$artifact_root/shape-m2/set-vision.ftsv" "$deployed_checkpoint"
    cp "$raw_head" "$deployed_head"
    ;;
  shape-m4)
    cp "$artifact_root/shape-m4/set-vision.ftsv" "$deployed_checkpoint"
    cp "$raw_head" "$deployed_head"
    ;;
  *) echo "selection produced invalid winner: $winner" >&2; exit 1 ;;
esac
"$project_root/tools/build-vision-extension.sh" >/dev/null
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
