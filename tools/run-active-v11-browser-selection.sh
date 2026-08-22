#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v11"
shortlist="$artifact_root/atlas-shortlist.tsv"
results="$artifact_root/browser-selection.tsv"
embedded_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
reference_checkpoint="$project_root/vision-artifacts/active-v10/sparse-p1-r050/set-vision.ftsv"

[[ -s "$shortlist" ]] || { echo "missing atlas shortlist: $shortlist" >&2; exit 2; }
[[ -s "$reference_checkpoint" ]] || { echo "missing active-v10 reference" >&2; exit 2; }

printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\tcheckpoint_sha256\n' >"$results"

candidates=(active-v10-reference)
while IFS= read -r candidate; do
  [[ "$candidate" == candidate || -z "$candidate" ]] && continue
  candidates+=("$candidate")
done <"$shortlist"

for candidate in "${candidates[@]}"; do
  if [[ "$candidate" == active-v10-reference ]]; then
    checkpoint="$reference_checkpoint"
  else
    checkpoint="$artifact_root/$candidate/set-vision.ftsv"
  fi
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint for $candidate" >&2; exit 2; }
  cp "$checkpoint" "$embedded_checkpoint"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  browser_log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$browser_log")"
  completed_boards=$(jq -r 'select(.event == "stream_summary") | .boards' "$browser_log" 2>/dev/null | tail -n 1 || true)
  if [[ "$completed_boards" != 512 ]]; then
    node "$project_root/tools/test-vision-extension.mjs" \
      --stream --audit --boards 512 --seed 2026082102 --report-every 512 >"$browser_log" 2>&1
  fi
  summary=$(jq -c 'select(.event == "stream_summary")' "$browser_log" | tail -n 1)
  [[ -n "$summary" ]] || { echo "missing browser summary for $candidate" >&2; exit 1; }
  board_accuracy=$(jq -r '.rawAccuracy' <<<"$summary")
  board_failures=$(jq -r '.failures' <<<"$summary")
  board_set_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
  board_no_set_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
  worst_conditional=$(awk -v has_acc="$board_set_accuracy" -v no_acc="$board_no_set_accuracy" \
    'BEGIN { if (has_acc < no_acc) print has_acc; else print no_acc }')
  checkpoint_sha256=$(shasum -a 256 "$checkpoint" | awk '{ print $1 }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$candidate" "$board_accuracy" "$board_failures" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$worst_conditional" "$checkpoint_sha256" >>"$results"
  printf '%s: failures=%s raw=%s has=%s no-set=%s\n' \
    "$candidate" "$board_failures" "$board_accuracy" "$board_set_accuracy" \
    "$board_no_set_accuracy"
done

winner=$(tail -n +2 "$results" | sort -t $'\t' -k3,3n -k6,6nr | head -n 1 | cut -f1)
[[ -n "$winner" ]] || { echo 'no browser-selection winner' >&2; exit 1; }
if [[ "$winner" == active-v10-reference ]]; then
  winner_checkpoint="$reference_checkpoint"
else
  winner_checkpoint="$artifact_root/$winner/set-vision.ftsv"
fi
cp "$winner_checkpoint" "$embedded_checkpoint"
"$project_root/tools/build-vision-extension.sh" >/dev/null
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
