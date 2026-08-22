#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v25"
deployed_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
reference_checkpoint="$artifact_root/deployed-reference-set-vision.ftsv"
reference_head="$artifact_root/frozen-board-head.json"
candidate_checkpoint="$artifact_root/step-250/set-vision.ftsv"
results="$artifact_root/browser-selection.tsv"

expected_reference_hash=ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec
actual_reference_hash=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
[[ "$actual_reference_hash" == "$expected_reference_hash" ]] || {
  echo "deployed checkpoint is not the fixed active-v12 reference: $actual_reference_hash" >&2
  exit 2
}
[[ -s "$candidate_checkpoint" ]] || { echo "missing candidate: $candidate_checkpoint" >&2; exit 2; }
cp "$deployed_checkpoint" "$reference_checkpoint"
cp "$deployed_head" "$reference_head"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' >"$results"

candidates=(deployed-reference step-250)
checkpoints=("$reference_checkpoint" "$candidate_checkpoint")
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  checkpoint=${checkpoints[$index]}
  cp "$checkpoint" "$deployed_checkpoint"
  cp "$reference_head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$log")"
  node "$project_root/tools/test-vision-extension.mjs" \
    --stream --audit --boards 512 --seed 2026090401 --report-every 512 >"$log" 2>&1
  summary=$(jq -c 'select(.event == "stream_summary")' "$log" | tail -n 1)
  board_accuracy=$(jq -r '.rawAccuracy' <<<"$summary")
  board_failures=$(jq -r '.failures' <<<"$summary")
  board_set_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
  board_no_set_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
  worst_conditional=$(awk -v has_acc="$board_set_accuracy" -v no_acc="$board_no_set_accuracy" \
    'BEGIN { if (has_acc < no_acc) print has_acc; else print no_acc }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$worst_conditional" >>"$results"
done

winner=$(awk -F '\t' '
  NR == 2 { reference_failures = $3; reference_has = $4; reference_no = $5 }
  NR == 3 {
    tolerance = 1 / 256
    if ($3 < reference_failures && $4 + tolerance >= reference_has && $5 + tolerance >= reference_no) {
      print $1
    } else {
      print "deployed-reference"
    }
  }
' "$results")

if [[ "$winner" == step-250 ]]; then
  cp "$candidate_checkpoint" "$deployed_checkpoint"
else
  cp "$reference_checkpoint" "$deployed_checkpoint"
fi
cp "$reference_head" "$deployed_head"
"$project_root/tools/build-vision-extension.sh" >/dev/null
column -t -s $'\t' "$results"
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
