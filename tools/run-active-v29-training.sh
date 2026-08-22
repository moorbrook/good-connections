#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v29"
atlas_dir="$artifact_root/training-atlases"
trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
data_seeds=(890021 990029 1090033 1190039)
init_seeds=(16045690984503098046 10723151780598845931 14313749767032793493 11400714819323198485)

node "$project_root/tools/verify-vision-corpus.mjs" "$atlas_dir" 1408 >/dev/null
(cd "$project_root" && npm run build >/dev/null)

train_replica() {
  local arm=$1
  local replica=$2
  local model_width=$3
  local model_height=$4
  local extra_flag=${5:-}
  local index=$((replica - 1))
  local output_dir="$artifact_root/$arm/replica-$replica"
  local checkpoint="$output_dir/set-vision.ftsv"
  mkdir -p "$output_dir"
  if [[ -s "$checkpoint" ]]; then
    printf '%s replica %s already complete\n' "$arm" "$replica"
    return
  fi

  local trainer_args=(
    --steps 20000
    --negative-logsumexp-loss-weight 2.5
    --atlas-dir "$atlas_dir"
    --data-seed "${data_seeds[$index]}"
    --init-seed "${init_seeds[$index]}"
    --curriculum-stdin
    --curriculum-rate 1
    --curriculum-examples-per-label 1
    --output "$checkpoint"
  )
  if [[ -n "$extra_flag" ]]; then trainer_args+=("$extra_flag"); fi

  set -o pipefail
  node "$project_root/tools/render-vision-boards.mjs" \
    --generate-boards 2048 \
    --seed 2026090801 \
    --model-width "$model_width" \
    --model-height "$model_height" \
    2>"$output_dir/render.log" \
  | "$trainer" "${trainer_args[@]}" \
    >"$output_dir/train.log" 2>&1

  local hash seconds parameters
  hash=$(shasum -a 256 "$checkpoint" | awk '{ print $1 }')
  seconds=$(awk '/^training_seconds:/ { print $2; exit }' "$output_dir/train.log")
  parameters=$(awk '/^policy_parameters:/ { print $2; exit }' "$output_dir/train.log")
  printf 'arm=%s replica=%s parameters=%s training_seconds=%s checkpoint_sha256=%s\n' \
    "$arm" "$replica" "$parameters" "$seconds" "$hash" | tee "$output_dir/summary.txt"
}

cargo build --manifest-path "$project_root/vision-mpsgraph/Cargo.toml" --release
for replica in 1 2 3 4; do train_replica low-resolution "$replica" 30 21; done

cargo build --manifest-path "$project_root/vision-mpsgraph/Cargo.toml" \
  --release --features high-resolution
for replica in 1 2 3 4; do train_replica high-resolution "$replica" 60 42; done

cargo build --manifest-path "$project_root/vision-mpsgraph/Cargo.toml" --release
train_replica wide-control 1 30 21 --wide-vision
