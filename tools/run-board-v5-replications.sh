#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
research_dir="$project_root/vision-artifacts/board-v5-research-atlases"
sealed_dir="$project_root/vision-artifacts/board-v5-sealed-atlases"
artifact_root="$project_root/vision-artifacts/autoresearch-v5"
results_file="$artifact_root/replications.tsv"
expected_research_digest=fb43f82a14ff3507267adf7ed754136b0fdc55ec5a30222cfe62cf36095625ac
expected_sealed_digest=5caf2e34172ac3171c74426c874524968181644c5be404738f7dd26103fe8fac

data_seeds=(99005 190001 290003 390007 490009 590011 690019 790021)
init_seeds=(
  11400714819323198485
  6364136223846793005
  1442695040888963407
  3202034522624059733
  3935559000370003845
  2862933555777941757
  7046029254386353131
  9650029242287828579
)

count_pngs() {
  find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}

if [[ $(count_pngs "$research_dir") != 240 ]]; then
  echo "Board-v5 research corpus must contain exactly 240 PNGs" >&2
  exit 2
fi
if [[ $(count_pngs "$sealed_dir") != 240 ]]; then
  echo "Board-v5 sealed corpus must contain exactly 240 PNGs before training" >&2
  exit 2
fi
research_digest=$("$project_root/tools/hash-vision-corpus.sh" "$research_dir")
sealed_digest=$("$project_root/tools/hash-vision-corpus.sh" "$sealed_dir")
if [[ "$research_digest" != "$expected_research_digest" ]]; then
  echo "Board-v5 research digest mismatch: $research_digest" >&2
  exit 2
fi
if [[ "$sealed_digest" != "$expected_sealed_digest" ]]; then
  echo "Board-v5 sealed digest mismatch: $sealed_digest" >&2
  exit 2
fi

mkdir -p "$artifact_root"
if [[ ! -e "$results_file" ]]; then
  printf 'run\tdata_seed\tinit_seed\tboard_accuracy\tboard_failures\ttriple_accuracy\tattribute_accuracy\ttraining_seconds\tsteps_per_second\tevaluation_seconds\n' >"$results_file"
fi

(cd "$project_root/vision-mpsgraph" && cargo build --release)
(cd "$project_root/vision" && cargo build --release)
mps_trainer="$project_root/vision-mpsgraph/target/release/set-vision-mpsgraph"
reference_evaluator="$project_root/vision/target/release/set-vision-trainer"

for index in "${!data_seeds[@]}"; do
  run=$((index + 1))
  printf -v tag 'replicate-%02d' "$run"
  output_dir="$artifact_root/$tag"
  log_file="$artifact_root/$tag.log"
  if awk -F '\t' -v run="$run" 'NR > 1 && $1 == run { found=1 } END { exit !found }' "$results_file"; then
    echo "Skipping completed $tag"
    continue
  fi
  if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "Partial artifacts already exist for $tag; inspect them before resuming" >&2
    exit 2
  fi
  mkdir -p "$output_dir"
  "$mps_trainer" \
    --steps 20000 \
    --no-augment \
    --data-seed "${data_seeds[$index]}" \
    --init-seed "${init_seeds[$index]}" \
    --output "$output_dir/set-vision.ftsv" \
    >"$log_file" 2>&1
  "$reference_evaluator" \
    --eval-only \
    --eval-batches 512 \
    --eval-boards 4096 \
    --validation-atlas-dir "$research_dir" \
    --output-dir "$output_dir" \
    >>"$log_file" 2>&1

  board_accuracy=$(awk '/^board_accuracy:/ { print $2 }' "$log_file")
  board_failures=$(awk '/^board_failures:/ { print $2 }' "$log_file")
  triple_accuracy=$(awk '/^triple_accuracy:/ { print $2 }' "$log_file")
  attribute_accuracy=$(awk '/^attribute_accuracy:/ { print $2 }' "$log_file")
  training_seconds=$(awk '/^training_seconds:/ { print $2 }' "$log_file")
  steps_per_second=$(awk '/^steps_per_second:/ { print $2 }' "$log_file")
  evaluation_seconds=$(awk '/^evaluation_seconds:/ { print $2 }' "$log_file")
  if [[ -z "$board_accuracy" || -z "$board_failures" || -z "$triple_accuracy" \
     || -z "$attribute_accuracy" || -z "$training_seconds" || -z "$steps_per_second" \
     || -z "$evaluation_seconds" ]]; then
    echo "Missing metrics from $tag" >&2
    tail -n 80 "$log_file" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run" "${data_seeds[$index]}" "${init_seeds[$index]}" "$board_accuracy" \
    "$board_failures" "$triple_accuracy" "$attribute_accuracy" "$training_seconds" \
    "$steps_per_second" "$evaluation_seconds" >>"$results_file"
  printf '%s: failures=%s triple=%s training=%ss evaluation=%ss\n' \
    "$tag" "$board_failures" "$triple_accuracy" "$training_seconds" "$evaluation_seconds"
done

selected_run=$(awk -F '\t' '
  NR > 1 {
    if (!seen || $5 < failures || ($5 == failures && $6 > triples) || ($5 == failures && $6 == triples && $1 < run)) {
      run=$1; failures=$5; triples=$6; seen=1
    }
  }
  END { if (seen) print run }
' "$results_file")
printf '%s\n' "$selected_run" >"$artifact_root/selected-run.txt"
printf 'Selected replicate %s by the locked board-v5 rule\n' "$selected_run"
