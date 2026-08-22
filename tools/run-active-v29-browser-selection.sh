#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v29"
model_dir="$project_root/vision-artifacts/model"
deployed_checkpoint="$model_dir/set-vision.ftsv"
deployed_head="$project_root/extension/board-head.json"
reference_checkpoint="$artifact_root/deployed-active-v12.ftsv"
reference_head="$artifact_root/frozen-active-v13-head.json"
results="$artifact_root/browser-selection.tsv"
finalize_only=false
if [[ ${1:-} == --finalize-only ]]; then
  finalize_only=true
elif (($# > 0)); then
  echo "usage: $0 [--finalize-only]" >&2
  exit 2
fi

if [[ "$finalize_only" == false ]]; then
  expected_checkpoint_hash=ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec
  expected_head_hash=993931b8fa4386d4e66ab71619b16d6c113672beb9387312addb8aed53ebde90
  actual_checkpoint_hash=$(shasum -a 256 "$deployed_checkpoint" | awk '{ print $1 }')
  actual_head_hash=$(shasum -a 256 "$deployed_head" | awk '{ print $1 }')
  [[ "$actual_checkpoint_hash" == "$expected_checkpoint_hash" ]] || { echo 'unexpected deployed checkpoint' >&2; exit 2; }
  [[ "$actual_head_hash" == "$expected_head_hash" ]] || { echo 'unexpected deployed board head' >&2; exit 2; }
  [[ ! -e "$results" ]] || { echo 'active-v29 browser selection already exists' >&2; exit 2; }
  cp "$deployed_checkpoint" "$reference_checkpoint"
  cp "$deployed_head" "$reference_head"
else
  [[ -s "$reference_checkpoint" && -s "$reference_head" ]] \
    || { echo 'missing frozen deployment backups for finalization' >&2; exit 2; }
  [[ -s "$results" && -s "$artifact_root/browser-paired.json" ]] \
    || { echo 'browser results are incomplete; refusing finalize-only' >&2; exit 2; }
fi

install_ensemble() {
  local arm=$1
  cp "$artifact_root/$arm/replica-1/set-vision.ftsv" "$model_dir/set-vision.ftsv"
  cp "$artifact_root/$arm/replica-2/set-vision.ftsv" "$model_dir/set-vision-ensemble-2.ftsv"
  cp "$artifact_root/$arm/replica-3/set-vision.ftsv" "$model_dir/set-vision-ensemble-3.ftsv"
  cp "$artifact_root/$arm/replica-4/set-vision.ftsv" "$model_dir/set-vision-ensemble-4.ftsv"
  cp "$reference_head" "$deployed_head"
  if [[ "$arm" == high-resolution ]]; then
    VISION_HIGH_RESOLUTION=1 VISION_ENSEMBLE_FOUR=1 \
      "$project_root/tools/build-vision-extension.sh" >/dev/null
  else
    VISION_ENSEMBLE_FOUR=1 "$project_root/tools/build-vision-extension.sh" >/dev/null
  fi
}

run_candidate() {
  local arm=$1
  local output_dir="$artifact_root/browser-$arm"
  [[ ! -e "$output_dir" ]] || { echo "browser output already exists: $output_dir" >&2; exit 2; }
  mkdir -p "$output_dir"
  install_ensemble "$arm"

  local pids=()
  local pending=0
  for start in 0 256 512 768 1024 1280 1536 1792; do
    node "$project_root/tools/test-vision-extension.mjs" \
      --stream \
      --boards 256 \
      --start "$start" \
      --seed 2026090891 \
      --report-every 256 \
      --focus-rate 0 \
      --curriculum "$output_dir/curriculum-$start.jsonl" \
      --evidence "$output_dir/evidence-$start.jsonl" \
      >"$output_dir/shard-$start.log" 2>&1 &
    pids+=("$!")
    pending=$((pending + 1))
    if ((pending == 4)); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
      pending=0
    fi
  done

  local combined="$output_dir/evidence.jsonl"
  : >"$combined"
  for start in 0 256 512 768 1024 1280 1536 1792; do
    cat "$output_dir/evidence-$start.jsonl" >>"$combined"
  done
  node "$project_root/tools/analyze-active-v15-attributes.mjs" \
    "$combined" "$output_dir/attributes.json" >"$output_dir/attributes-summary.json"
}

if [[ "$finalize_only" == false ]]; then
  run_candidate low-resolution
  run_candidate high-resolution

  printf 'candidate\tboards\traw_accuracy\tfailures\thas_set_accuracy\thas_set_failures\tno_set_accuracy\tno_set_failures\tguarded_accuracy\tguarded_coverage\tshape_accuracy\tfill_accuracy\n' >"$results"
fi
record_candidate() {
  local arm=$1
  local output_dir="$artifact_root/browser-$arm"
  local failures=0 has_correct=0 no_correct=0 accepted=0 accepted_correct=0
  for start in 0 256 512 768 1024 1280 1536 1792; do
    local summary shard_failures has_accuracy no_accuracy coverage guarded_accuracy
    local shard_has_correct shard_no_correct shard_accepted shard_accepted_correct
    summary=$(jq -c 'select(.event == "stream_summary")' "$output_dir/shard-$start.log" | tail -n 1)
    shard_failures=$(jq -r '.failures' <<<"$summary")
    has_accuracy=$(jq -r '.hasSetAccuracy' <<<"$summary")
    no_accuracy=$(jq -r '.noSetAccuracy' <<<"$summary")
    coverage=$(jq -r '.guardedCoverage' <<<"$summary")
    guarded_accuracy=$(jq -r '.guardedAccuracy // 0' <<<"$summary")
    shard_has_correct=$(awk -v value="$has_accuracy" 'BEGIN { printf "%.0f", value * 128 }')
    shard_no_correct=$(awk -v value="$no_accuracy" 'BEGIN { printf "%.0f", value * 128 }')
    shard_accepted=$(awk -v value="$coverage" 'BEGIN { printf "%.0f", value * 256 }')
    shard_accepted_correct=$(awk -v value="$guarded_accuracy" -v count="$shard_accepted" \
      'BEGIN { printf "%.0f", value * count }')
    failures=$((failures + shard_failures))
    has_correct=$((has_correct + shard_has_correct))
    no_correct=$((no_correct + shard_no_correct))
    accepted=$((accepted + shard_accepted))
    accepted_correct=$((accepted_correct + shard_accepted_correct))
  done
  local raw_accuracy has_accuracy no_accuracy guarded_accuracy guarded_coverage shape fill
  raw_accuracy=$(awk -v value="$failures" 'BEGIN { print (2048 - value) / 2048 }')
  has_accuracy=$(awk -v value="$has_correct" 'BEGIN { print value / 1024 }')
  no_accuracy=$(awk -v value="$no_correct" 'BEGIN { print value / 1024 }')
  guarded_accuracy=$(awk -v value="$accepted_correct" -v count="$accepted" \
    'BEGIN { if (count == 0) print 0; else print value / count }')
  guarded_coverage=$(awk -v value="$accepted" 'BEGIN { print value / 2048 }')
  shape=$(jq -r '.overall.attributes[] | select(.name == "shape") | .accuracy' "$output_dir/attributes.json")
  fill=$(jq -r '.overall.attributes[] | select(.name == "fill") | .accuracy' "$output_dir/attributes.json")
  printf '%s\t2048\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$raw_accuracy" "$failures" "$has_accuracy" "$((1024 - has_correct))" \
    "$no_accuracy" "$((1024 - no_correct))" "$guarded_accuracy" "$guarded_coverage" \
    "$shape" "$fill" >>"$results"
}

if [[ "$finalize_only" == false ]]; then
  record_candidate low-resolution
  record_candidate high-resolution
  node "$project_root/tools/compare-active-v29-paired.mjs" \
    "$artifact_root/browser-low-resolution" \
    "$artifact_root/browser-high-resolution" \
    2048 >"$artifact_root/browser-paired.json"
fi

selected=$(awk -F '\t' '
  $1 == "low-resolution" {
    low_raw=$3; low_has_failures=$6; low_no_failures=$8; low_guarded=$9; low_coverage=$10
  }
  $1 == "high-resolution" {
    high_raw=$3; high_has_failures=$6; high_no_failures=$8; high_guarded=$9; high_coverage=$10
    shape=$11; fill=$12
  }
  END {
    raw_improved = high_raw > low_raw
    guard_tripwire = high_guarded > low_guarded && !raw_improved
    if (raw_improved && high_has_failures <= low_has_failures && high_no_failures <= low_no_failures &&
        shape >= 0.995 && fill >= 0.995 && high_coverage >= low_coverage && !guard_tripwire)
      print "high-resolution"
    else print "low-resolution"
  }
' "$results")
paired_pass=$(jq -r '.passesTwoStandardErrors' "$artifact_root/browser-paired.json")
if [[ "$paired_pass" != true ]]; then selected=low-resolution; fi

if [[ "$selected" == high-resolution ]]; then
  install_ensemble high-resolution
  node "$project_root/tools/test-vision-extension.mjs" >"$artifact_root/browser-smoke.log" 2>&1
else
  cp "$reference_checkpoint" "$deployed_checkpoint"
  cp "$reference_head" "$deployed_head"
  "$project_root/tools/build-vision-extension.sh" >/dev/null
fi
column -t -s $'\t' "$results"
cat "$artifact_root/browser-paired.json"
printf 'selected_candidate=%s\n' "$selected" | tee "$artifact_root/browser-selection.txt"
