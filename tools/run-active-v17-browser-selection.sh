#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v17"
results="$artifact_root/browser-selection.tsv"
mkdir -p "$artifact_root"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' > "$results"

for candidate in baseline four-view; do
  log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$log")"
  completed_boards=$(jq -r 'select(.event == "stream_summary") | .boards' "$log" 2>/dev/null | tail -n 1 || true)
  if [[ "$completed_boards" != 512 ]]; then
    if [[ "$candidate" == four-view ]]; then
      node "$project_root/tools/test-vision-extension.mjs" \
        --stream --audit --boards 512 --seed 2026082701 --report-every 512 --multi-view > "$log" 2>&1
    else
      node "$project_root/tools/test-vision-extension.mjs" \
        --stream --audit --boards 512 --seed 2026082701 --report-every 512 > "$log" 2>&1
    fi
  fi
  summary=$(jq -c 'select(.event == "stream_summary")' "$log" | tail -n 1)
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
  NR == 2 {
    baseline_failures = $3
    baseline_has = $4
    baseline_no = $5
  }
  $1 == "four-view" {
    tolerance = 1 / 256
    if ($3 < baseline_failures && $4 + tolerance >= baseline_has && $5 + tolerance >= baseline_no) {
      winner = "four-view"
    } else {
      winner = "baseline"
    }
  }
  END { print winner }
' "$results")
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
