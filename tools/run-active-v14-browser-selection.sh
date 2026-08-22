#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v14"
deployed_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
results="$artifact_root/browser-selection.tsv"
raw_head="$artifact_root/raw-board-head.json"
reference="$artifact_root/reference-set-vision.ftsv"

expected_reference_hash=ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec
actual_reference_hash=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
[[ "$actual_reference_hash" == "$expected_reference_hash" ]] || {
  echo "deployed checkpoint is not the fixed active-v12 reference: $actual_reference_hash" >&2
  exit 2
}
cp "$deployed_checkpoint" "$reference"
printf '{\n  "type": "score-threshold",\n  "threshold": 0\n}\n' > "$raw_head"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' > "$results"

candidates=(reference browser-r050 browser-r100)
checkpoints=(
  "$reference"
  "$artifact_root/browser-r050/set-vision.ftsv"
  "$artifact_root/browser-r100/set-vision.ftsv"
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
      --stream --audit --boards 512 --seed 2026082402 --report-every 512 > "$browser_log" 2>&1
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
  NR == 2 { reference_has = $4; reference_no = $5 }
  NR > 1 {
    tolerance = 1 / 256
    eligible = ($1 == "reference") || ($4 + tolerance >= reference_has && $5 + tolerance >= reference_no)
    priority = ($1 == "browser-r050") ? 0 : (($1 == "reference") ? 1 : 2)
    if (eligible && (!selected || $3 < failures || ($3 == failures && ($6 > worst || ($6 == worst && priority < best_priority))))) {
      selected = 1
      candidate = $1
      failures = $3
      worst = $6
      best_priority = priority
    }
  }
  END { print candidate }
' "$results")
case "$winner" in
  reference) selected_checkpoint="$reference" ;;
  browser-r050) selected_checkpoint="$artifact_root/browser-r050/set-vision.ftsv" ;;
  browser-r100) selected_checkpoint="$artifact_root/browser-r100/set-vision.ftsv" ;;
  *) echo "selection produced invalid winner: $winner" >&2; exit 1 ;;
esac
cp "$selected_checkpoint" "$deployed_checkpoint"
cp "$raw_head" "$deployed_head"
"$project_root/tools/build-vision-extension.sh" >/dev/null
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
