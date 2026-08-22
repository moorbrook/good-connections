#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v12"
results="$artifact_root/browser-selection.tsv"
embedded_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"

candidates=(soup-w050 soup-w025 active-v10-reference active-v11-targeted)
atlas_failures=(98 101 110 115)
checkpoints=(
  "$artifact_root/soup-w050/set-vision.ftsv"
  "$artifact_root/soup-w025/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v10/sparse-p1-r050/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v11/targeted-r100-p2/set-vision.ftsv"
)

mkdir -p "$artifact_root"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\tatlas_failures\tcheckpoint_sha256\n' >"$results"

for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  checkpoint=${checkpoints[$index]}
  atlas_failure_count=${atlas_failures[$index]}
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint for $candidate" >&2; exit 2; }
  cp "$checkpoint" "$embedded_checkpoint"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  browser_log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$browser_log")"
  completed_boards=$(jq -r 'select(.event == "stream_summary") | .boards' "$browser_log" 2>/dev/null | tail -n 1 || true)
  if [[ "$completed_boards" != 512 ]]; then
    node "$project_root/tools/test-vision-extension.mjs" \
      --stream --audit --boards 512 --seed 2026082201 --report-every 512 >"$browser_log" 2>&1
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$candidate" "$board_accuracy" "$board_failures" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$worst_conditional" "$atlas_failure_count" \
    "$checkpoint_sha256" >>"$results"
  printf '%s: failures=%s raw=%s has=%s no-set=%s\n' \
    "$candidate" "$board_failures" "$board_accuracy" "$board_set_accuracy" \
    "$board_no_set_accuracy"
done

winner=$(tail -n +2 "$results" | sort -t $'\t' -k3,3n -k6,6nr -k7,7n | head -n 1 | cut -f1)
[[ -n "$winner" ]] || { echo 'no browser-selection winner' >&2; exit 1; }
for index in "${!candidates[@]}"; do
  if [[ "${candidates[$index]}" == "$winner" ]]; then
    cp "${checkpoints[$index]}" "$embedded_checkpoint"
    break
  fi
done
"$project_root/tools/build-vision-extension.sh" >/dev/null
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
