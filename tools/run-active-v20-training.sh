#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v20"
atlas_dir="$project_root/vision-artifacts/board-v8-training-atlases"
trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"

[[ -x "$trainer" ]] || { echo "missing MPSGraph trainer: $trainer" >&2; exit 2; }
[[ -d "$atlas_dir" ]] || { echo "missing training atlases: $atlas_dir" >&2; exit 2; }

train_candidate() {
  local groups=$1
  local output_dir="$artifact_root/groups-$groups"
  local checkpoint="$output_dir/set-vision.ftsv"
  mkdir -p "$output_dir"
  if [[ ! -s "$checkpoint" ]]; then
    "$trainer" \
      --steps 10000 \
      --negative-logsumexp-loss-weight 2.5 \
      --board-correlated-negatives \
      --negative-board-groups "$groups" \
      --atlas-dir "$atlas_dir" \
      --data-seed 590011 \
      --init-seed 3935559000370003845 \
      --output "$checkpoint" >"$output_dir/train.log" 2>&1
  fi
  local hash
  local seconds
  hash=$(shasum -a 256 "$checkpoint" | awk '{ print $1 }')
  seconds=$(awk '/^training_seconds:/ { print $2; exit }' "$output_dir/train.log")
  printf 'groups-%s checkpoint_sha256=%s training_seconds=%s\n' "$groups" "$hash" "$seconds"
}

train_candidate 4 &
pid_four=$!
train_candidate 8 &
pid_eight=$!
wait "$pid_four"
wait "$pid_eight"
