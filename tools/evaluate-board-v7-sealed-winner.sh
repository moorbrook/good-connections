#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
  echo 'Usage: tools/evaluate-board-v7-sealed-winner.sh PRIMARY_DIR SECONDARY_DIR REP2_DIR REP3_DIR OUTPUT_ROOT' >&2
  exit 2
fi

primary_dir=$1
secondary_dir=$2
rep2_dir=$3
rep3_dir=$4
output_root=$5
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
sealed_dir="$project_root/vision-artifacts/board-v7-sealed-atlases"
digest_file="$project_root/vision/BOARD_V7_CORPORA.sha256"

checkpoint_dirs=("$primary_dir" "$secondary_dir" "$rep2_dir" "$rep3_dir")
for checkpoint_dir in "${checkpoint_dirs[@]}"; do
  [[ -f "$checkpoint_dir/set-vision.ftsv" ]] || {
    echo "missing checkpoint: $checkpoint_dir/set-vision.ftsv" >&2
    exit 1
  }
done
[[ -d "$sealed_dir" ]] || { echo "missing sealed corpus: $sealed_dir" >&2; exit 1; }
expected_digest=$(awk '$1 == "sealed" && $2 == 516 { print $3 }' "$digest_file")
actual_digest=$("$project_root/tools/hash-vision-corpus.sh" "$sealed_dir")
[[ -n "$expected_digest" && "$actual_digest" == "$expected_digest" ]] || {
  echo 'sealed corpus digest mismatch' >&2
  exit 1
}
if [[ -e "$output_root" ]] && [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be new or empty: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"

shard_seeds=(16001 16002 16003)
triple_seeds=(177145488695121 177145488695122 177145488695123)
board_seeds=(194222009835345 194222009835346 194222009835347)
eval_batches=(342 341 341)
eval_boards=(3334 3333 3333)
weights=(0.44 0.36 0.10 0.10)
shard_root=$(mktemp -d "${TMPDIR:-/tmp}/board-v7-sealed-shards.XXXXXX")
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
  done < <(find -H "$sealed_dir" -maxdepth 1 -type f -name "*-$seed-*.png" -print0 | sort -z)
  shard_count=$(find "$shard" -maxdepth 1 -type l | wc -l | tr -d ' ')
  [[ "$shard_count" == 172 ]] || {
    echo "seed $seed has $shard_count atlases, expected 172" >&2
    exit 1
  }

  names=(primary secondary third fourth)
  output_dirs=()
  for model_index in "${!checkpoint_dirs[@]}"; do
    model_output="$shard_output/${names[$model_index]}"
    mkdir -p "$model_output"
    cp "${checkpoint_dirs[$model_index]}/set-vision.ftsv" "$model_output/set-vision.ftsv"
    output_dirs+=("$model_output")
  done

  log="$shard_output/eval.log"
  "$evaluator" \
    --eval-only \
    --eval-batches "${eval_batches[$index]}" \
    --eval-boards "${eval_boards[$index]}" \
    --triple-seed "${triple_seeds[$index]}" \
    --board-seed "${board_seeds[$index]}" \
    --ensemble-output-dir "${output_dirs[1]}" \
    --ensemble-third-output-dir "${output_dirs[2]}" \
    --ensemble-fourth-output-dir "${output_dirs[3]}" \
    --ensemble-four-weights "${weights[@]}" \
    --validation-atlas-dir "$shard" \
    --output-dir "${output_dirs[0]}" > "$log"

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

[[ "$total_triple" == 65536 && "$total_boards" == 10000 ]] || {
  echo "aggregate count mismatch: triples=$total_triple boards=$total_boards" >&2
  exit 1
}
triple_accuracy=$(awk -v correct="$total_triple_correct" -v total="$total_triple" 'BEGIN { printf "%.9f", correct / total }')
board_accuracy=$(awk -v failed="$total_board_failures" -v total="$total_boards" 'BEGIN { printf "%.9f", (total - failed) / total }')
printf 'aggregate triple_correct=%s/%s triple_accuracy=%s board_failures=%s/%s board_accuracy=%s\n' \
  "$total_triple_correct" "$total_triple" "$triple_accuracy" \
  "$total_board_failures" "$total_boards" "$board_accuracy" | tee "$output_root/aggregate.txt"
