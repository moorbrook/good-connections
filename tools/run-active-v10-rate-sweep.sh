#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v10"
curriculum="$artifact_root/development-curriculum.jsonl"
training_atlases="$project_root/vision-artifacts/board-v8-training-atlases"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
results="$artifact_root/rate-sweep.tsv"
mps_trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
evaluator="$project_root/vision/target/release/set-vision-trainer"

[[ -s "$curriculum" ]] || { echo "missing development curriculum: $curriculum" >&2; exit 2; }
[[ -d "$training_atlases" ]] || { echo "missing training atlases: $training_atlases" >&2; exit 2; }
[[ -d "$research_atlases" ]] || { echo "missing research atlases: $research_atlases" >&2; exit 2; }

mkdir -p "$artifact_root"
(cd "$project_root/vision-mpsgraph" && cargo build --release)
(cd "$project_root/vision" && cargo build --release)

printf 'rate\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\ttriple_accuracy\tattribute_accuracy\ttraining_seconds\tcheckpoint_sha256\n' >"$results"

rates=(0 0.02 0.05 0.10 0.20)
for rate in "${rates[@]}"; do
  tag="rate-${rate/./p}"
  output_dir="$artifact_root/$tag"
  train_log="$output_dir/train.log"
  eval_log="$output_dir/eval.log"
  checkpoint="$output_dir/set-vision.ftsv"
  mkdir -p "$output_dir"

  if [[ ! -s "$checkpoint" ]]; then
    if [[ "$rate" == 0 ]]; then
      "$mps_trainer" \
        --steps 10000 \
        --negative-logsumexp-loss-weight 2.5 \
        --atlas-dir "$training_atlases" \
        --data-seed 590011 \
        --init-seed 3935559000370003845 \
        --output "$checkpoint" >"$train_log" 2>&1
    else
      set -o pipefail
      node "$project_root/tools/render-vision-curriculum.mjs" \
        --curriculum "$curriculum" \
      | "$mps_trainer" \
        --steps 10000 \
        --negative-logsumexp-loss-weight 2.5 \
        --atlas-dir "$training_atlases" \
        --data-seed 590011 \
        --init-seed 3935559000370003845 \
        --curriculum-stdin \
        --curriculum-rate "$rate" \
        --output "$checkpoint" >"$train_log" 2>&1
    fi
  fi

  if [[ ! -s "$eval_log" ]]; then
    "$evaluator" \
      --eval-only \
      --mixed-boards \
      --eval-batches 256 \
      --eval-boards 4096 \
      --triple-seed 327145488695021 \
      --board-seed 324222009835245 \
      --validation-atlas-dir "$research_atlases" \
      --output-dir "$output_dir" >"$eval_log" 2>&1
  fi

  board_accuracy=$(awk '/^board_accuracy:/ { print $2; exit }' "$eval_log")
  board_failures=$(awk '/^board_failures:/ { print $2; exit }' "$eval_log")
  board_set_accuracy=$(awk '/^board_set_accuracy:/ { print $2; exit }' "$eval_log")
  board_no_set_accuracy=$(awk '/^board_no_set_accuracy:/ { print $2; exit }' "$eval_log")
  triple_accuracy=$(awk '/^triple_accuracy:/ { print $2; exit }' "$eval_log")
  attribute_accuracy=$(awk '/^attribute_accuracy:/ { print $2; exit }' "$eval_log")
  training_seconds=$(awk '/^training_seconds:/ { print $2; exit }' "$train_log")
  checkpoint_sha256=$(shasum -a 256 "$checkpoint" | awk '{ print $1 }')
  for value in "$board_accuracy" "$board_failures" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$triple_accuracy" "$attribute_accuracy" "$training_seconds"; do
    [[ -n "$value" ]] || { echo "missing metric for $tag" >&2; exit 1; }
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rate" "$board_accuracy" "$board_failures" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$triple_accuracy" "$attribute_accuracy" "$training_seconds" \
    "$checkpoint_sha256" >>"$results"
  printf '%s: failures=%s raw=%s has=%s no-set=%s triple=%s\n' \
    "$tag" "$board_failures" "$board_accuracy" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$triple_accuracy"
done

winner=$(awk -F '\t' '
  NR > 1 {
    if (!seen || $3 < failures || ($3 == failures && $6 > triples)) {
      rate=$1; failures=$3; triples=$6; seen=1
    }
  }
  END { if (seen) print rate }
' "$results")
[[ -n "$winner" ]] || { echo 'no rate-sweep winner' >&2; exit 1; }
printf 'selected_rate=%s\n' "$winner" | tee "$artifact_root/selection.txt"
