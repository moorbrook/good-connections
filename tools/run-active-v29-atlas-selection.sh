#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v29"
atlas_dir="$artifact_root/eligibility-atlases"
evaluator="$project_root/vision/target/release/set-vision-trainer"
results="$artifact_root/atlas-selection.tsv"

node "$project_root/tools/verify-vision-corpus.mjs" "$atlas_dir" 704 >/dev/null

evaluate() {
  local tag=$1
  local primary=$2
  shift 2
  local log="$artifact_root/$tag-atlas.log"
  "$evaluator" \
    --eval-only \
    --mixed-boards \
    --eval-batches 256 \
    --eval-boards 4096 \
    --triple-seed 327145488795021 \
    --board-seed 324222009935245 \
    --validation-atlas-dir "$atlas_dir" \
    --output-dir "$primary" \
    "$@" >"$log" 2>&1
}

evaluate_arm() {
  local arm=$1
  local features=$2
  if [[ "$features" == high-resolution ]]; then
    cargo build --manifest-path "$project_root/vision/Cargo.toml" \
      --release --features high-resolution --bin set-vision-trainer
  else
    cargo build --manifest-path "$project_root/vision/Cargo.toml" \
      --release --bin set-vision-trainer
  fi
  for replica in 1 2 3 4; do
    evaluate "$arm-replica-$replica" "$artifact_root/$arm/replica-$replica"
  done
  evaluate "$arm-ensemble" "$artifact_root/$arm/replica-1" \
    --ensemble-output-dir "$artifact_root/$arm/replica-2" \
    --ensemble-third-output-dir "$artifact_root/$arm/replica-3" \
    --ensemble-fourth-output-dir "$artifact_root/$arm/replica-4" \
    --ensemble-four-weights 1 1 1 1
}

evaluate_arm low-resolution default
evaluate_arm high-resolution high-resolution

cargo build --manifest-path "$project_root/vision/Cargo.toml" \
  --release --bin set-vision-trainer
evaluate wide-control "$artifact_root/wide-control/replica-1" --wide-vision

printf 'candidate\tboard_accuracy\tboard_failures\thas_set_accuracy\thas_set_failures\tno_set_accuracy\tno_set_failures\ttriple_accuracy\tattribute_accuracy\n' >"$results"
record() {
  local tag=$1
  local log="$artifact_root/$tag-atlas.log"
  local accuracy failures has_accuracy no_accuracy has_correct no_correct triple attribute
  accuracy=$(awk '/^board_accuracy:/ { print $2; exit }' "$log")
  failures=$(awk '/^board_failures:/ { print $2; exit }' "$log")
  has_accuracy=$(awk '/^board_set_accuracy:/ { print $2; exit }' "$log")
  no_accuracy=$(awk '/^board_no_set_accuracy:/ { print $2; exit }' "$log")
  has_correct=$(awk '/^board_set_accuracy:/ { value=$3; gsub(/[()]/, "", value); split(value, parts, "\/"); print parts[1]; exit }' "$log")
  no_correct=$(awk '/^board_no_set_accuracy:/ { value=$3; gsub(/[()]/, "", value); split(value, parts, "\/"); print parts[1]; exit }' "$log")
  triple=$(awk '/^triple_accuracy:/ { print $2; exit }' "$log")
  attribute=$(awk '/^attribute_accuracy:/ { print $2; exit }' "$log")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$accuracy" "$failures" "$has_accuracy" "$((2048 - has_correct))" \
    "$no_accuracy" "$((2048 - no_correct))" "$triple" "$attribute" >>"$results"
}

for arm in low-resolution high-resolution; do
  for replica in 1 2 3 4; do record "$arm-replica-$replica"; done
  record "$arm-ensemble"
done
record wide-control

selection=$(awk -F '\t' '
  $1 == "low-resolution-ensemble" { low_total=$3; low_has=$5; low_no=$7 }
  $1 == "high-resolution-ensemble" {
    high_total=$3; high_has=$5; high_no=$7
    if (high_total < low_total && high_has <= low_has && high_no <= low_no) selected="high-resolution-ensemble"
    else selected="low-resolution-ensemble"
  }
  END { print selected }
' "$results")
column -t -s $'\t' "$results"
printf 'selected_candidate=%s\n' "$selection" | tee "$artifact_root/atlas-selection.txt"
