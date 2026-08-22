#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v22"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
evaluator="$project_root/vision/target/release/set-vision-trainer"
control_dir="$project_root/vision-artifacts/active-v10/rate-0"
results="$artifact_root/atlas-selection.tsv"

[[ -x "$evaluator" ]] || { echo "missing evaluator: $evaluator" >&2; exit 2; }
[[ -d "$research_atlases" ]] || { echo "missing research atlases: $research_atlases" >&2; exit 2; }

evaluate_candidate() {
  local tag=$1
  local output_dir="$artifact_root/$tag"
  [[ -s "$output_dir/set-vision.ftsv" ]] || { echo "missing checkpoint: $output_dir" >&2; exit 2; }
  if [[ ! -s "$output_dir/atlas-eval.log" ]]; then
    "$evaluator" \
      --eval-only \
      --mixed-boards \
      --eval-batches 256 \
      --eval-boards 4096 \
      --triple-seed 327145488695021 \
      --board-seed 324222009835245 \
      --validation-atlas-dir "$research_atlases" \
      --output-dir "$output_dir" >"$output_dir/atlas-eval.log" 2>&1
  fi
}

evaluate_candidate count-4 &
pid_four=$!
evaluate_candidate count-8 &
pid_eight=$!
wait "$pid_four"
wait "$pid_eight"

printf 'candidate\tboard_accuracy\tboard_failures\thas_set_accuracy\thas_set_failures\tno_set_accuracy\tno_set_failures\ttriple_accuracy\teligible\n' >"$results"

record_result() {
  local tag=$1
  local log=$2
  local accuracy failures has_accuracy no_accuracy has_successes no_successes has_failures no_failures eligible triple
  accuracy=$(awk '/^board_accuracy:/ { print $2; exit }' "$log")
  failures=$(awk '/^board_failures:/ { print $2; exit }' "$log")
  has_accuracy=$(awk '/^board_set_accuracy:/ { print $2; exit }' "$log")
  no_accuracy=$(awk '/^board_no_set_accuracy:/ { print $2; exit }' "$log")
  has_successes=$(awk '/^board_set_accuracy:/ { print $3; exit }' "$log" | sed -E 's/^\(([0-9]+)\/2048\)$/\1/')
  no_successes=$(awk '/^board_no_set_accuracy:/ { print $3; exit }' "$log" | sed -E 's/^\(([0-9]+)\/2048\)$/\1/')
  has_failures=$((2048 - has_successes))
  no_failures=$((2048 - no_successes))
  triple=$(awk '/^triple_accuracy:/ { print $2; exit }' "$log")
  eligible=no
  if [[ "$tag" != control && "$failures" -lt 117 && "$has_failures" -le 53 && "$no_failures" -le 72 ]]; then
    eligible=yes
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$accuracy" "$failures" "$has_accuracy" "$has_failures" \
    "$no_accuracy" "$no_failures" "$triple" "$eligible" >>"$results"
}

record_result control "$control_dir/eval.log"
record_result count-4 "$artifact_root/count-4/atlas-eval.log"
record_result count-8 "$artifact_root/count-8/atlas-eval.log"
column -t -s $'\t' "$results"

winner=$(awk -F '\t' '
  NR > 1 && $9 == "yes" {
    if (!seen || $3 < failures || ($3 == failures && $8 > triples)) {
      candidate=$1; failures=$3; triples=$8; seen=1
    }
  }
  END { if (seen) print candidate }
' "$results")
printf 'selected_candidate=%s\n' "${winner:-control}" | tee "$artifact_root/atlas-selection.txt"
