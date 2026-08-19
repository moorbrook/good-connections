#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
tag=${1:-}
description=${2:-}
if [[ -z "$tag" || -z "$description" ]]; then
  echo 'Usage: tools/run-vision-experiment.sh TAG DESCRIPTION [training flags...]' >&2
  exit 2
fi
shift 2

if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'TAG may contain only letters, digits, dots, underscores, and hyphens' >&2
  exit 2
fi

artifact_root="$project_root/vision-artifacts/autoresearch"
output_dir="$artifact_root/$tag"
log_file="$artifact_root/$tag.log"
results_file="$artifact_root/results-board-v3.tsv"
if [[ -e "$output_dir" || -e "$log_file" ]]; then
  echo "Experiment artifacts already exist for tag: $tag" >&2
  exit 2
fi

mkdir -p "$artifact_root"
if [[ ! -e "$results_file" ]]; then
  printf 'tag\tboard_accuracy\tboard_failures\ttriple_accuracy\tattribute_accuracy\tevaluation_seconds\tstatus\tdescription\n' >"$results_file"
fi

(cd "$project_root/vision" && cargo build --release)
status=0
"$project_root/vision/target/release/set-vision-trainer" \
  "$@" \
  --eval-batches 512 \
  --eval-boards 4096 \
  --validation-atlas-dir "$project_root/vision-artifacts/validation-multiraster-atlases" \
  --output-dir "$output_dir" \
  >"$log_file" 2>&1 || status=$?

if ((status != 0)); then
  printf '%s\t0\t4096\t0\t0\t0\tcrash\t%s\n' "$tag" "$description" >>"$results_file"
  tail -n 40 "$log_file" >&2
  exit "$status"
fi

board_accuracy=$(awk '/^board_accuracy:/ { print $2 }' "$log_file")
board_failures=$(awk '/^board_failures:/ { print $2 }' "$log_file")
triple_accuracy=$(awk '/^triple_accuracy:/ { print $2 }' "$log_file")
attribute_accuracy=$(awk '/^attribute_accuracy:/ { print $2 }' "$log_file")
evaluation_seconds=$(awk '/^evaluation_seconds:/ { print $2 }' "$log_file")
if [[ -z "$board_accuracy" || -z "$board_failures" || -z "$triple_accuracy" || -z "$attribute_accuracy" || -z "$evaluation_seconds" ]]; then
  printf '%s\t0\t4096\t0\t0\t0\tcrash\tmissing evaluator output: %s\n' "$tag" "$description" >>"$results_file"
  tail -n 40 "$log_file" >&2
  exit 1
fi

best=$(awk -F '\t' 'NR > 1 && $7 == "keep" { if (!seen || $3 < failures || ($3 == failures && $4 > triples)) { failures=$3; triples=$4; seen=1 } } END { if (seen) print failures "\t" triples }' "$results_file")
run_status=keep
if [[ -n "$best" ]]; then
  best_failures=$(printf '%s' "$best" | awk -F '\t' '{ print $1 }')
  best_triples=$(printf '%s' "$best" | awk -F '\t' '{ print $2 }')
  run_status=$(awk -v failures="$board_failures" -v triples="$triple_accuracy" -v best_failures="$best_failures" -v best_triples="$best_triples" 'BEGIN { if (failures < best_failures || (failures == best_failures && triples > best_triples)) print "keep"; else print "discard" }')
fi

clean_description=${description//$'\t'/ }
clean_description=${clean_description//$'\n'/ }
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$tag" "$board_accuracy" "$board_failures" "$triple_accuracy" "$attribute_accuracy" \
  "$evaluation_seconds" "$run_status" "$clean_description" >>"$results_file"

printf 'tag: %s\nboard_accuracy: %s\nboard_failures: %s\ntriple_accuracy: %s\nattribute_accuracy: %s\nevaluation_seconds: %s\nstatus: %s\nlog: %s\n' \
  "$tag" "$board_accuracy" "$board_failures" "$triple_accuracy" "$attribute_accuracy" \
  "$evaluation_seconds" "$run_status" "$log_file"
