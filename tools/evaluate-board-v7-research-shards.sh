#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 && $# != 3 && $# != 4 && $# != 5 ]]; then
  echo 'Usage: tools/evaluate-board-v7-research-shards.sh CHECKPOINT_DIR OUTPUT_ROOT [ENSEMBLE_CHECKPOINT_DIR [THIRD_CHECKPOINT_DIR [FOURTH_CHECKPOINT_DIR]]]' >&2
  exit 2
fi

checkpoint_dir=$1
output_root=$2
ensemble_checkpoint_dir=${3:-}
third_checkpoint_dir=${4:-}
fourth_checkpoint_dir=${5:-}
ensemble_primary_weight=${ENSEMBLE_PRIMARY_WEIGHT:-0.5}
read -r -a ensemble_weights <<< "${ENSEMBLE_WEIGHTS:-0.495 0.405 0.100}"
read -r -a ensemble_four_weights <<< "${ENSEMBLE_FOUR_WEIGHTS:-0.4675 0.3825 0.075 0.075}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
research_dir="$project_root/vision-artifacts/board-v7-research-atlases"
checkpoint="$checkpoint_dir/set-vision.ftsv"
digest_file="$project_root/vision/BOARD_V7_CORPORA.sha256"

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
if [[ -n "$ensemble_checkpoint_dir" ]]; then
  ensemble_checkpoint="$ensemble_checkpoint_dir/set-vision.ftsv"
  [[ -f "$ensemble_checkpoint" ]] || { echo "missing ensemble checkpoint: $ensemble_checkpoint" >&2; exit 1; }
fi
if [[ -n "$third_checkpoint_dir" ]]; then
  third_checkpoint="$third_checkpoint_dir/set-vision.ftsv"
  [[ -f "$third_checkpoint" ]] || { echo "missing third checkpoint: $third_checkpoint" >&2; exit 1; }
fi
if [[ -n "$fourth_checkpoint_dir" ]]; then
  fourth_checkpoint="$fourth_checkpoint_dir/set-vision.ftsv"
  [[ -f "$fourth_checkpoint" ]] || { echo "missing fourth checkpoint: $fourth_checkpoint" >&2; exit 1; }
fi
[[ -d "$research_dir" ]] || { echo "missing research corpus: $research_dir" >&2; exit 1; }
expected_digest=$(awk '$1 == "research" && $2 == 516 { print $3 }' "$digest_file")
actual_digest=$("$project_root/tools/hash-vision-corpus.sh" "$research_dir")
[[ -n "$expected_digest" && "$actual_digest" == "$expected_digest" ]] || {
  echo "research corpus digest mismatch" >&2
  exit 1
}

if [[ -e "$output_root" ]] && [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be new or empty: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"

# Each process loads one 172-atlas shard. Keep the processes sequential: the
# point of this helper is a bounded-memory protocol, not parallel evaluation.
shard_seeds=(15001 15002 15003)
triple_seeds=(177145488695021 177145488695022 177145488695023)
board_seeds=(194222009835245 194222009835246 194222009835247)
eval_batches=(342 341 341)
eval_boards=(2731 2731 2730)
shard_root=$(mktemp -d "${TMPDIR:-/tmp}/board-v7-shards.XXXXXX")
trap 'rm -rf -- "$shard_root"' EXIT

(cd "$project_root/vision" && cargo build --release --quiet)
evaluator="$project_root/vision/target/release/set-vision-trainer"
results="$output_root/results.tsv"
printf 'seed\ttriple_seed\tboard_seed\teval_batches\teval_boards\ttriple_correct\ttriple_total\tboard_failures\tboard_total\tlog\n' > "$results"

total_triple_correct=0
total_triple=0
total_board_failures=0
total_boards=0
for index in "${!shard_seeds[@]}"; do
  seed=${shard_seeds[$index]}
  shard="$shard_root/seed-$seed"
  shard_output="$output_root/seed-$seed"
  mkdir -p "$shard" "$shard_output"
  while IFS= read -r -d '' atlas; do
    ln -s "$atlas" "$shard/$(basename "$atlas")"
  done < <(find "$research_dir" -maxdepth 1 -type f -name "*-$seed-*.png" -print0 | sort -z)
  shard_count=$(find "$shard" -maxdepth 1 -type l | wc -l | tr -d ' ')
  [[ "$shard_count" == 172 ]] || { echo "seed $seed has $shard_count atlases, expected 172" >&2; exit 1; }
  cp "$checkpoint" "$shard_output/set-vision.ftsv"
  ensemble_args=()
  if [[ -n "$ensemble_checkpoint_dir" ]]; then
    ensemble_output="$shard_output/ensemble"
    mkdir -p "$ensemble_output"
    cp "$ensemble_checkpoint" "$ensemble_output/set-vision.ftsv"
    if [[ -n "$third_checkpoint_dir" ]]; then
      third_output="$shard_output/third"
      mkdir -p "$third_output"
      cp "$third_checkpoint" "$third_output/set-vision.ftsv"
      if [[ -n "$fourth_checkpoint_dir" ]]; then
        fourth_output="$shard_output/fourth"
        mkdir -p "$fourth_output"
        cp "$fourth_checkpoint" "$fourth_output/set-vision.ftsv"
        ensemble_args=(
          --ensemble-output-dir "$ensemble_output"
          --ensemble-third-output-dir "$third_output"
          --ensemble-fourth-output-dir "$fourth_output"
          --ensemble-four-weights "${ensemble_four_weights[@]}"
        )
      else
        ensemble_args=(
          --ensemble-output-dir "$ensemble_output"
          --ensemble-third-output-dir "$third_output"
          --ensemble-weights "${ensemble_weights[@]}"
        )
      fi
    else
      ensemble_args=(
        --ensemble-output-dir "$ensemble_output"
        --ensemble-primary-weight "$ensemble_primary_weight"
      )
    fi
  fi
  log="$shard_output/eval.log"
  "$evaluator" \
    --eval-only \
    --eval-batches "${eval_batches[$index]}" \
    --eval-boards "${eval_boards[$index]}" \
    --triple-seed "${triple_seeds[$index]}" \
    --board-seed "${board_seeds[$index]}" \
    "${ensemble_args[@]}" \
    --validation-atlas-dir "$shard" \
    --output-dir "$shard_output" > "$log"
  triple_correct=$(awk -F '[ :/]+' '/^triple_correct:/ { print $2; exit }' "$log")
  triple_total=$(awk -F '[ :/]+' '/^triple_correct:/ { print $3; exit }' "$log")
  board_failures=$(awk -F '[ :/]+' '/^board_failures:/ { print $2; exit }' "$log")
  [[ "$triple_correct" =~ ^[0-9]+$ && "$triple_total" =~ ^[0-9]+$ && "$board_failures" =~ ^[0-9]+$ ]] || {
    echo "could not parse evaluator totals for seed $seed; see $log" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "${triple_seeds[$index]}" "${board_seeds[$index]}" \
    "${eval_batches[$index]}" "${eval_boards[$index]}" "$triple_correct" "$triple_total" \
    "$board_failures" "${eval_boards[$index]}" "$log" >> "$results"
  total_triple_correct=$((total_triple_correct + triple_correct))
  total_triple=$((total_triple + triple_total))
  total_board_failures=$((total_board_failures + board_failures))
  total_boards=$((total_boards + eval_boards[$index]))
done

expected_triple=$((1024 * 64))
[[ "$total_triple" == "$expected_triple" && "$total_boards" == 8192 ]] || {
  echo "aggregate totals are ${total_triple} triples and ${total_boards} boards; expected ${expected_triple} and 8192" >&2
  exit 1
}
triple_accuracy=$(awk -v correct="$total_triple_correct" -v total="$total_triple" 'BEGIN { printf "%.9f", correct / total }')
board_accuracy=$(awk -v failures="$total_board_failures" -v total="$total_boards" 'BEGIN { printf "%.9f", (total - failures) / total }')
printf 'aggregate triple_correct=%s/%s triple_accuracy=%s board_failures=%s/%s board_accuracy=%s\n' \
  "$total_triple_correct" "$total_triple" "$triple_accuracy" "$total_board_failures" "$total_boards" "$board_accuracy" \
  | tee "$output_root/aggregate.txt"
