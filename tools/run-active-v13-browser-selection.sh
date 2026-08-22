#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v13"
selected_head="$artifact_root/selected-board-head.json"
raw_head="$artifact_root/raw-board-head.json"
deployed_head="$project_root/extension/board-head.json"
results="$artifact_root/browser-selection.tsv"

[[ -s "$selected_head" ]] || { echo "missing fitted head: $selected_head" >&2; exit 2; }
mkdir -p "$artifact_root"
printf '{\n  "type": "score-threshold",\n  "threshold": 0\n}\n' >"$raw_head"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' >"$results"

candidates=(raw fitted)
configs=("$raw_head" "$selected_head")
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  cp "${configs[$index]}" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  browser_log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$browser_log")"
  completed_boards=$(jq -r 'select(.event == "stream_summary") | .boards' "$browser_log" 2>/dev/null | tail -n 1 || true)
  if [[ "$completed_boards" != 512 ]]; then
    node "$project_root/tools/test-vision-extension.mjs" \
      --stream --audit --boards 512 --seed 2026082302 --report-every 512 >"$browser_log" 2>&1
  fi
  summary=$(jq -c 'select(.event == "stream_summary")' "$browser_log" | tail -n 1)
  board_accuracy=$(jq -r '.rawAccuracy' <<<"$summary")
  board_failures=$(jq -r '.failures' <<<"$summary")
  board_set_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
  board_no_set_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
  worst_conditional=$(awk -v has_acc="$board_set_accuracy" -v no_acc="$board_no_set_accuracy" \
    'BEGIN { if (has_acc < no_acc) print has_acc; else print no_acc }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$worst_conditional" >>"$results"
done

raw_failures=$(awk -F '\t' '$1 == "raw" { print $3 }' "$results")
fitted_failures=$(awk -F '\t' '$1 == "fitted" { print $3 }' "$results")
raw_has=$(awk -F '\t' '$1 == "raw" { print $4 }' "$results")
fitted_has=$(awk -F '\t' '$1 == "fitted" { print $4 }' "$results")
raw_no=$(awk -F '\t' '$1 == "raw" { print $5 }' "$results")
fitted_no=$(awk -F '\t' '$1 == "fitted" { print $5 }' "$results")
winner=$(awk -v raw_failures="$raw_failures" -v fitted_failures="$fitted_failures" \
  -v raw_has="$raw_has" -v fitted_has="$fitted_has" -v raw_no="$raw_no" -v fitted_no="$fitted_no" \
  'BEGIN {
    tolerance = 1 / 256
    if (fitted_failures < raw_failures && fitted_has + tolerance >= raw_has && fitted_no + tolerance >= raw_no) print "fitted"
    else print "raw"
  }')
if [[ "$winner" == fitted ]]; then
  cp "$selected_head" "$deployed_head"
else
  cp "$raw_head" "$deployed_head"
fi
"$project_root/tools/build-vision-extension.sh" >/dev/null
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
