#!/usr/bin/env bash
set -u -o pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v14"
renderer="$project_root/tools/render-vision-boards.mjs"
trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
atlas_dir="$project_root/vision-artifacts/board-v8-training-atlases"
temporary_root=$(mktemp -d)
fifo_r050="$temporary_root/browser-r050.fifo"
fifo_r100="$temporary_root/browser-r100.fifo"
pid_r050=
pid_r100=

cleanup() {
  if [[ -n "$pid_r050" ]]; then kill "$pid_r050" 2>/dev/null || true; fi
  if [[ -n "$pid_r100" ]]; then kill "$pid_r100" 2>/dev/null || true; fi
  rm -r "$temporary_root"
}
trap cleanup EXIT HUP INT TERM

[[ -x "$trainer" ]] || { echo "missing MPSGraph trainer: $trainer" >&2; exit 2; }
[[ -d "$atlas_dir" ]] || { echo "missing training atlases: $atlas_dir" >&2; exit 2; }
mkdir -p "$artifact_root/browser-r050" "$artifact_root/browser-r100"
mkfifo "$fifo_r050" "$fifo_r100"

train_candidate() {
  local candidate=$1
  local rate=$2
  local input_fifo=$3
  "$trainer" \
    --steps 10000 \
    --negative-logsumexp-loss-weight 2.5 \
    --atlas-dir "$atlas_dir" \
    --data-seed 590011 \
    --init-seed 3935559000370003845 \
    --browser-training-stdin \
    --browser-batch-rate "$rate" \
    --output "$artifact_root/$candidate/set-vision.ftsv" \
    < "$input_fifo" > "$artifact_root/$candidate/train.log" 2>&1
}

train_candidate browser-r050 0.5 "$fifo_r050" &
pid_r050=$!
train_candidate browser-r100 1.0 "$fifo_r100" &
pid_r100=$!

node "$renderer" --generate-boards 2048 --seed 2026082401 \
  2> "$artifact_root/render.log" \
  | tee "$fifo_r050" > "$fifo_r100"
pipeline_status=("${PIPESTATUS[@]}")

wait "$pid_r050"
status_r050=$?
pid_r050=
wait "$pid_r100"
status_r100=$?
pid_r100=

if [[ ${pipeline_status[0]} -ne 0 || ${pipeline_status[1]} -ne 0 \
  || $status_r050 -ne 0 || $status_r100 -ne 0 ]]; then
  printf 'active-v14 training failed: renderer=%s tee=%s r050=%s r100=%s\n' \
    "${pipeline_status[0]}" "${pipeline_status[1]}" "$status_r050" "$status_r100" >&2
  exit 1
fi

for candidate in browser-r050 browser-r100; do
  checkpoint="$artifact_root/$candidate/set-vision.ftsv"
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
  hash=$(shasum -a 256 "$checkpoint" | awk '{ print $1 }')
  seconds=$(awk '/^training_seconds:/ { print $2; exit }' "$artifact_root/$candidate/train.log")
  printf '%s checkpoint_sha256=%s training_seconds=%s\n' "$candidate" "$hash" "$seconds"
done
