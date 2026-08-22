#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v27"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
evaluator="$project_root/vision/target/release/set-vision-trainer"
reference_log="$project_root/vision-artifacts/active-v26/deployed-atlas-eval.log"
results="$artifact_root/atlas-selection.tsv"

cargo build --manifest-path "$project_root/vision/Cargo.toml" --release --bin set-vision-trainer

evaluate_snapshot() {
  local step=$1
  local output_dir="$artifact_root/step-$step"
  [[ -s "$output_dir/set-vision.ftsv" ]] || { echo "missing checkpoint: $output_dir" >&2; exit 2; }
  "$evaluator" \
    --eval-only \
    --mixed-boards \
    --eval-batches 256 \
    --eval-boards 4096 \
    --triple-seed 327145488695021 \
    --board-seed 324222009835245 \
    --validation-atlas-dir "$research_atlases" \
    --output-dir "$output_dir" >"$output_dir/atlas-eval.log" 2>&1
}

evaluate_snapshot 250 &
pid_250=$!
evaluate_snapshot 1000 &
pid_1000=$!
wait "$pid_250"
wait "$pid_1000"

printf 'candidate\tboard_accuracy\tboard_failures\thas_set_accuracy\thas_set_failures\tno_set_accuracy\tno_set_failures\ttriple_accuracy\tattribute_accuracy\teligible\n' >"$results"

record_result() {
  local tag=$1
  local log=$2
  local accuracy failures has_accuracy no_accuracy has_successes no_successes has_failures no_failures triple attribute eligible
  accuracy=$(awk '/^board_accuracy:/ { print $2; exit }' "$log")
  failures=$(awk '/^board_failures:/ { print $2; exit }' "$log")
  has_accuracy=$(awk '/^board_set_accuracy:/ { print $2; exit }' "$log")
  no_accuracy=$(awk '/^board_no_set_accuracy:/ { print $2; exit }' "$log")
  has_successes=$(awk '/^board_set_accuracy:/ { print $3; exit }' "$log" | sed -E 's/^\(([0-9]+)\/2048\)$/\1/')
  no_successes=$(awk '/^board_no_set_accuracy:/ { print $3; exit }' "$log" | sed -E 's/^\(([0-9]+)\/2048\)$/\1/')
  has_failures=$((2048 - has_successes))
  no_failures=$((2048 - no_successes))
  triple=$(awk '/^triple_accuracy:/ { print $2; exit }' "$log")
  attribute=$(awk '/^attribute_accuracy:/ { print $2; exit }' "$log")
  eligible=no
  if [[ "$tag" != deployed-reference && "$failures" -lt 98 && "$has_failures" -le 45 && "$no_failures" -le 61 ]]; then
    eligible=yes
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$accuracy" "$failures" "$has_accuracy" "$has_failures" \
    "$no_accuracy" "$no_failures" "$triple" "$attribute" "$eligible" >>"$results"
}

record_result deployed-reference "$reference_log"
record_result step-250 "$artifact_root/step-250/atlas-eval.log"
record_result step-1000 "$artifact_root/step-1000/atlas-eval.log"
column -t -s $'\t' "$results"

winner=$(awk -F '\t' '
  NR > 1 && $10 == "yes" {
    if (!seen || $3 < failures || ($3 == failures && $8 > triple)) {
      candidate=$1; failures=$3; triple=$8; seen=1
    }
  }
  END { if (seen) print candidate }
' "$results")
printf 'selected_candidate=%s\n' "${winner:-deployed-reference}" | tee "$artifact_root/atlas-selection.txt"
