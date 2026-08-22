#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v29/parity"
atlas_dir="$project_root/vision-artifacts/board-v8-training-atlases"
mps_trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
native_evaluator="$project_root/vision/target/release/set-vision-trainer"

mkdir -p "$artifact_root/checkpoint"
cargo build --manifest-path "$project_root/vision-mpsgraph/Cargo.toml" \
  --release --features high-resolution
cargo build --manifest-path "$project_root/vision/Cargo.toml" \
  --release --features high-resolution --bin set-vision-trainer

"$mps_trainer" \
  --steps 1 \
  --negative-logsumexp-loss-weight 2.5 \
  --no-augment \
  --atlas-dir "$atlas_dir" \
  --data-seed 99005 \
  --init-seed 3935559000370003845 \
  --output "$artifact_root/checkpoint/set-vision.ftsv" \
  >"$artifact_root/mpsgraph.log" 2>&1

"$native_evaluator" \
  --probe-only \
  --negative-logsumexp-loss-weight 2.5 \
  --attribute-loss-weight 4 \
  --atlas-dir "$atlas_dir" \
  --validation-atlas-dir "$atlas_dir" \
  --output-dir "$artifact_root/checkpoint" \
  >"$artifact_root/frankentorch.log" 2>&1

mps_loss=$(awk '/^final_probe_loss:/ { print $2; exit }' "$artifact_root/mpsgraph.log")
native_loss=$(awk '/^probe_loss:/ { print $2; exit }' "$artifact_root/frankentorch.log")
awk -v left="$mps_loss" -v right="$native_loss" 'BEGIN {
  delta = left - right; if (delta < 0) delta = -delta;
  if (delta > 0.00001) exit 1;
}' || { echo "loss parity failed: MPSGraph=$mps_loss FrankenTorch=$native_loss" >&2; exit 1; }

normalize_gradient() {
  sed -n 's/^probe_shared_weight_gradient_0: \[\(.*\)\]$/\1/p' "$1" | tr -d ','
}
mps_gradient=$(normalize_gradient "$artifact_root/mpsgraph.log")
native_gradient=$(normalize_gradient "$artifact_root/frankentorch.log")
awk -v left="$mps_gradient" -v right="$native_gradient" 'BEGIN {
  left_count = split(left, left_values, " ");
  right_count = split(right, right_values, " ");
  if (left_count != 8 || right_count != 8) exit 2;
  for (i = 1; i <= 8; i++) {
    delta = left_values[i] - right_values[i];
    if (delta < 0) delta = -delta;
    if (delta > 0.00001) exit 1;
  }
}' || {
  echo "gradient parity failed: MPSGraph=[$mps_gradient] FrankenTorch=[$native_gradient]" >&2
  exit 1
}

parameters=$(awk '/^policy_parameters:/ { print $2; exit }' "$artifact_root/mpsgraph.log")
printf 'input=60x42 parameters=%s mpsgraph_loss=%s frankentorch_loss=%s parity=pass\n' \
  "$parameters" "$mps_loss" "$native_loss" | tee "$artifact_root/summary.txt"
