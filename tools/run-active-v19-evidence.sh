#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v19"
evaluator="$project_root/vision/target/release/evaluate_rendered_boards"
checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
mkdir -p "$artifact_root"

(cd "$project_root/vision" && cargo build --release --bin evaluate_rendered_boards)

pids=""
for offset in 0 4096 8192 12288; do
  shard="$artifact_root/shard-$offset"
  mkdir -p "$shard"
  completed=0
  if [[ -f "$shard/evidence.jsonl" ]]; then
    completed=$(wc -l < "$shard/evidence.jsonl" | tr -d ' ')
  fi
  if [[ "$completed" == 4096 ]]; then
    continue
  fi
  (
    set -o pipefail
    node "$project_root/tools/render-vision-boards.mjs" \
      --generate-boards 4096 --seed 2026082901 --start "$offset" 2> "$shard/render.log" \
    | "$evaluator" --checkpoint "$checkpoint" --index-offset "$offset" \
      > "$shard/evidence.jsonl" 2> "$shard/eval.log"
  ) &
  pids="$pids $!"
done
for pid in $pids; do wait "$pid"; done

for offset in 0 4096 8192 12288; do
  shard="$artifact_root/shard-$offset"
  lines=$(wc -l < "$shard/evidence.jsonl" | tr -d ' ')
  [[ "$lines" == 4096 ]] || { echo "shard $offset has $lines evidence rows" >&2; exit 1; }
  summary=$(tail -n 1 "$shard/eval.log")
  [[ "$summary" == rendered_board_summary\ boards=4096* ]] || {
    echo "shard $offset has no completed evaluator summary" >&2
    exit 1
  }
  printf 'offset=%s %s\n' "$offset" "$summary"
done
