#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v28"
reference="$project_root/vision-artifacts/model/set-vision.ftsv"
adapted="$project_root/vision-artifacts/active-v27/step-250/set-vision.ftsv"
reference_log="$project_root/vision-artifacts/active-v26/deployed-atlas-eval.log"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
blender="$project_root/vision/target/release/blend_checkpoints"
scope_checker="$project_root/vision/target/release/check_checkpoint_scope"
evaluator="$project_root/vision/target/release/set-vision-trainer"
results="$artifact_root/atlas-selection.tsv"

mkdir -p "$artifact_root"
cargo build --manifest-path "$project_root/vision/Cargo.toml" --release \
  --bin blend_checkpoints --bin check_checkpoint_scope --bin set-vision-trainer

for weight in 025 050 075; do
  output_dir="$artifact_root/w$weight"
  mkdir -p "$output_dir"
  primary_weight=$(awk -v value="$weight" 'BEGIN { print value / 100 }')
  "$blender" "$adapted" "$reference" "$primary_weight" "$output_dir/set-vision.ftsv" \
    >"$output_dir/blend.log"
  "$scope_checker" "$reference" "$output_dir/set-vision.ftsv" \
    --allow-prefix attribute_head_2. \
    --allow-prefix attribute_head_3. \
    --allow-prefix fill_texture_head. \
    --allow-prefix attribute_relation_ \
    --allow-prefix set_relation_ \
    >"$output_dir/scope.log"
done

evaluate_candidate() {
  local weight=$1
  local output_dir="$artifact_root/w$weight"
  "$evaluator" \
    --eval-only --mixed-boards --eval-batches 256 --eval-boards 4096 \
    --triple-seed 327145488695021 --board-seed 324222009835245 \
    --validation-atlas-dir "$research_atlases" --output-dir "$output_dir" \
    >"$output_dir/atlas-eval.log" 2>&1
}

pids=()
for weight in 025 050 075; do
  evaluate_candidate "$weight" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

printf 'candidate\tboard_accuracy\tboard_failures\thas_set_failures\tno_set_failures\ttriple_accuracy\tattribute_accuracy\teligible\n' >"$results"
record_result() {
  local tag=$1
  local log=$2
  local accuracy failures has_successes no_successes has_failures no_failures triple attribute eligible
  accuracy=$(awk '/^board_accuracy:/ { print $2; exit }' "$log")
  failures=$(awk '/^board_failures:/ { print $2; exit }' "$log")
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$accuracy" "$failures" "$has_failures" "$no_failures" "$triple" "$attribute" "$eligible" >>"$results"
}

record_result deployed-reference "$reference_log"
for weight in 025 050 075; do record_result "w$weight" "$artifact_root/w$weight/atlas-eval.log"; done
column -t -s $'\t' "$results"

winner=$(awk -F '\t' '
  NR > 1 && $8 == "yes" {
    weight = substr($1, 2) + 0
    if (!seen || $3 < failures || ($3 == failures && ($6 > triple || ($6 == triple && weight < best_weight)))) {
      candidate=$1; failures=$3; triple=$6; best_weight=weight; seen=1
    }
  }
  END { if (seen) print candidate }
' "$results")
printf 'selected_candidate=%s\n' "${winner:-deployed-reference}" | tee "$artifact_root/atlas-selection.txt"

for weight in 025 050 075; do shasum -a 256 "$artifact_root/w$weight/set-vision.ftsv"; done
