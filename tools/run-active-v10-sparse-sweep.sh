#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v10"
curriculum="$artifact_root/development-curriculum.jsonl"
training_atlases="$project_root/vision-artifacts/board-v8-training-atlases"
research_atlases="$project_root/vision-artifacts/board-v8-research-atlases"
results="$artifact_root/sparse-sweep.tsv"
mps_trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
evaluator="$project_root/vision/target/release/set-vision-trainer"

[[ -s "$curriculum" ]] || { echo "missing development curriculum: $curriculum" >&2; exit 2; }
[[ -s "$artifact_root/rate-0/set-vision.ftsv" ]] || { echo 'missing rate-zero control' >&2; exit 2; }
[[ -d "$training_atlases" ]] || { echo "missing training atlases: $training_atlases" >&2; exit 2; }
[[ -d "$research_atlases" ]] || { echo "missing research atlases: $research_atlases" >&2; exit 2; }

(cd "$project_root/vision-mpsgraph" && cargo build --release)
(cd "$project_root/vision" && cargo build --release)

printf 'candidate\trate\texamples_per_label\tboard_accuracy\tboard_failures\tboard_set_accuracy\tboard_no_set_accuracy\ttriple_accuracy\tattribute_accuracy\ttraining_seconds\tcheckpoint_sha256\n' >"$results"
baseline=$(awk -F '\t' 'NR == 2 { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8 "\t" $9 }' "$artifact_root/rate-sweep.tsv")
printf 'control\t0\t0\t%s\n' "$baseline" >>"$results"

candidates=(p1-r025 p1-r050 p1-r100 p2-r100)
rates=(0.25 0.50 1.00 1.00)
examples=(1 1 1 2)
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  rate=${rates[$index]}
  example_count=${examples[$index]}
  output_dir="$artifact_root/sparse-$candidate"
  train_log="$output_dir/train.log"
  eval_log="$output_dir/eval.log"
  checkpoint="$output_dir/set-vision.ftsv"
  mkdir -p "$output_dir"

  if [[ ! -s "$checkpoint" ]]; then
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
      --curriculum-examples-per-label "$example_count" \
      --output "$checkpoint" >"$train_log" 2>&1
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
    [[ -n "$value" ]] || { echo "missing metric for $candidate" >&2; exit 1; }
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$candidate" "$rate" "$example_count" "$board_accuracy" "$board_failures" \
    "$board_set_accuracy" "$board_no_set_accuracy" "$triple_accuracy" "$attribute_accuracy" \
    "$training_seconds" "$checkpoint_sha256" >>"$results"
  printf '%s: failures=%s raw=%s has=%s no-set=%s triple=%s\n' \
    "$candidate" "$board_failures" "$board_accuracy" "$board_set_accuracy" \
    "$board_no_set_accuracy" "$triple_accuracy"
done

winner=$(awk -F '\t' '
  NR > 1 {
    if (!seen || $5 < failures || ($5 == failures && $8 > triples)) {
      candidate=$1; failures=$5; triples=$8; seen=1
    }
  }
  END { if (seen) print candidate }
' "$results")
[[ -n "$winner" ]] || { echo 'no sparse-sweep winner' >&2; exit 1; }
printf 'selected_candidate=%s\n' "$winner" | tee "$artifact_root/sparse-selection.txt"
