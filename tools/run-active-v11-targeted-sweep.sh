#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v11"
curriculum="$project_root/vision-artifacts/active-v11-curriculum.jsonl"
training_atlases="$project_root/vision-artifacts/board-v8-training-atlases"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
results="$artifact_root/targeted-sweep.tsv"
mps_trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
evaluator="$project_root/vision/target/release/set-vision-trainer"

[[ -s "$curriculum" ]] || { echo "missing curriculum: $curriculum" >&2; exit 2; }
[[ -d "$training_atlases" ]] || { echo "missing training atlases: $training_atlases" >&2; exit 2; }
[[ -d "$research_atlases" ]] || { echo "missing research atlases: $research_atlases" >&2; exit 2; }

mkdir -p "$artifact_root"
(cd "$project_root/vision-mpsgraph" && cargo build --release)
(cd "$project_root/vision" && cargo build --release)

printf 'candidate\tmode\trate\texamples_per_label\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\ttriple_accuracy\tattribute_accuracy\ttraining_seconds\tcheckpoint_sha256\n' >"$results"

candidates=(balanced-r050-p1 targeted-r050-p1 targeted-r100-p1 targeted-r100-p2)
modes=(balanced targeted targeted targeted)
rates=(0.50 0.50 1.00 1.00)
examples=(1 1 1 2)
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  mode=${modes[$index]}
  rate=${rates[$index]}
  example_count=${examples[$index]}
  output_dir="$artifact_root/$candidate"
  train_log="$output_dir/train.log"
  eval_log="$output_dir/eval.log"
  checkpoint="$output_dir/set-vision.ftsv"
  mkdir -p "$output_dir"

  trainer_args=(
    --steps 10000
    --negative-logsumexp-loss-weight 2.5
    --atlas-dir "$training_atlases"
    --data-seed 590011
    --init-seed 3935559000370003845
    --curriculum-stdin
    --curriculum-rate "$rate"
    --curriculum-examples-per-label "$example_count"
  )
  if [[ "$mode" == targeted ]]; then
    trainer_args+=(--targeted-curriculum)
  fi
  trainer_args+=(--output "$checkpoint")
  node "$project_root/tools/render-vision-boards.mjs" \
    --curriculum "$curriculum" \
  | "$mps_trainer" "${trainer_args[@]}" >"$train_log" 2>&1

  "$evaluator" \
    --eval-only \
    --mixed-boards \
    --eval-batches 256 \
    --eval-boards 4096 \
    --triple-seed 327145488695021 \
    --board-seed 324222009835245 \
    --validation-atlas-dir "$research_atlases" \
    --output-dir "$output_dir" >"$eval_log" 2>&1

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
    [[ -n "$value" ]] || { echo "missing metric for $candidate" >&2; exit 1; }
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$candidate" "$mode" "$rate" "$example_count" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$triple_accuracy" "$attribute_accuracy" \
    "$training_seconds" "$checkpoint_sha256" >>"$results"
  printf '%s: failures=%s raw=%s has=%s no-set=%s triple=%s\n' \
    "$candidate" "$board_failures" "$board_accuracy" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$triple_accuracy"
done

{
  printf 'candidate\n'
  tail -n +2 "$results" | sort -t $'\t' -k6,6n -k9,9nr | head -n 3 | cut -f1
} >"$artifact_root/atlas-shortlist.tsv"
