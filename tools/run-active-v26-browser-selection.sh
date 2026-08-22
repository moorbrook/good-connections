#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v26"
deployed_checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
reference_head="$artifact_root/deployed-reference-board-head.json"
candidate_head="$artifact_root/candidate-board-head.json"
results="$artifact_root/browser-selection.tsv"

expected_checkpoint_hash=ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec
expected_reference_head_hash=993931b8fa4386d4e66ab71619b16d6c113672beb9387312addb8aed53ebde90
actual_checkpoint_hash=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
actual_head_hash=$(shasum -a 256 "$deployed_head" | awk '{ print $1 }')
[[ "$actual_checkpoint_hash" == "$expected_checkpoint_hash" ]] || {
  echo "deployed checkpoint is not the fixed active-v12 reference: $actual_checkpoint_hash" >&2
  exit 2
}
[[ "$actual_head_hash" == "$expected_reference_head_hash" ]] || {
  echo "deployed head is not the fixed active-v13 reference: $actual_head_hash" >&2
  exit 2
}
[[ -s "$candidate_head" ]] || { echo "missing candidate: $candidate_head" >&2; exit 2; }

cp "$deployed_head" "$reference_head"
printf 'candidate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\n' >"$results"

candidates=(deployed-reference recall-constrained)
heads=("$reference_head" "$candidate_head")
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  head=${heads[$index]}
  cp "$head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
  log="$artifact_root/$candidate/browser-development.log"
  mkdir -p "$(dirname "$log")"
  node "$project_root/tools/test-vision-extension.mjs" \
    --stream --audit --boards 512 --seed 2026090501 --report-every 512 >"$log" 2>&1
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

if [[ "$winner" == recall-constrained ]]; then
  cp "$candidate_head" "$deployed_head"
else
  cp "$reference_head" "$deployed_head"
fi
"$project_root/tools/build-vision-extension.sh" >/dev/null
column -t -s $'\t' "$results"
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/browser-selection.txt"
