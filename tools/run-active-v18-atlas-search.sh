#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v18"
control="$project_root/vision-artifacts/active-v10/rate-0/set-vision.ftsv"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
blender="$project_root/vision/target/release/blend_checkpoints"
evaluator="$project_root/vision/target/release/set-vision-trainer"
results="$artifact_root/atlas-results.tsv"

mkdir -p "$artifact_root"
(cd "$project_root/vision" && cargo build --release --bin blend_checkpoints --bin set-vision-trainer)

tags=(m2-w025 m2-w050 m2-w075 m4-w025 m4-w050 m4-w075)
shape_checkpoints=(
  "$project_root/vision-artifacts/active-v16/shape-m2/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v16/shape-m2/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v16/shape-m2/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v16/shape-m4/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v16/shape-m4/set-vision.ftsv"
  "$project_root/vision-artifacts/active-v16/shape-m4/set-vision.ftsv"
)
weights=(0.25 0.50 0.75 0.25 0.50 0.75)

for index in "${!tags[@]}"; do
  output_dir="$artifact_root/${tags[$index]}"
  checkpoint="$output_dir/set-vision.ftsv"
  mkdir -p "$output_dir"
  if [[ ! -s "$checkpoint" ]]; then
    "$blender" "${shape_checkpoints[$index]}" "$control" "${weights[$index]}" "$checkpoint" \
      > "$output_dir/blend.log"
  fi
done

for batch_start in 0 2 4; do
  pids=""
  for offset in 0 1; do
    index=$((batch_start + offset))
    output_dir="$artifact_root/${tags[$index]}"
    eval_log="$output_dir/eval.log"
    if [[ ! -s "$eval_log" ]]; then
      "$evaluator" \
        --eval-only \
        --mixed-boards \
        --eval-batches 256 \
        --eval-boards 4096 \
        --triple-seed 327145488695021 \
        --board-seed 324222009835245 \
        --validation-atlas-dir "$research_atlases" \
        --output-dir "$output_dir" > "$eval_log" 2>&1 &
      pids="$pids $!"
    fi
  done
  for pid in $pids; do wait "$pid"; done
done

printf 'candidate\tshape_multiplier\tshape_weight\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\tworst_conditional\ttriple_accuracy\tattribute_accuracy\tcheckpoint_sha256\n' > "$results"
for index in "${!tags[@]}"; do
  tag=${tags[$index]}
  output_dir="$artifact_root/$tag"
  eval_log="$output_dir/eval.log"
  board_accuracy=$(awk '/^board_accuracy:/ { print $2; exit }' "$eval_log")
  board_failures=$(awk '/^board_failures:/ { print $2; exit }' "$eval_log")
  board_set_accuracy=$(awk '/^board_set_accuracy:/ { print $2; exit }' "$eval_log")
  board_no_set_accuracy=$(awk '/^board_no_set_accuracy:/ { print $2; exit }' "$eval_log")
  triple_accuracy=$(awk '/^triple_accuracy:/ { print $2; exit }' "$eval_log")
  attribute_accuracy=$(awk '/^attribute_accuracy:/ { print $2; exit }' "$eval_log")
  worst_conditional=$(awk -v has_acc="$board_set_accuracy" -v no_acc="$board_no_set_accuracy" \
    'BEGIN { if (has_acc < no_acc) print has_acc; else print no_acc }')
  checkpoint_sha256=$(shasum -a 256 "$output_dir/set-vision.ftsv" | awk '{ print $1 }')
  shape_multiplier=${tag:1:1}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$shape_multiplier" "${weights[$index]}" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$worst_conditional" "$triple_accuracy" \
    "$attribute_accuracy" "$checkpoint_sha256" >> "$results"
done

cat "$results"
