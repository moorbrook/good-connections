#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v27"
checkpoint="$project_root/vision-artifacts/model/set-vision.ftsv"
curriculum="$project_root/vision-artifacts/active-v26/audit.log"
trainer="$project_root/vision/target/release/train_board_policy"
scope_checker="$project_root/vision/target/release/check_checkpoint_scope"

expected_checkpoint_hash=ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec
actual_checkpoint_hash=$(shasum -a 256 "$checkpoint" | awk '{ print $1 }')
[[ "$actual_checkpoint_hash" == "$expected_checkpoint_hash" ]] || {
  echo "deployed checkpoint is not active-v12: $actual_checkpoint_hash" >&2
  exit 2
}
[[ -s "$curriculum" ]] || { echo "missing retired v26 audit curriculum" >&2; exit 2; }
[[ ! -e "$artifact_root/train.log" ]] || { echo "active-v27 training already exists" >&2; exit 2; }

mkdir -p "$artifact_root"
(cd "$project_root" && npm run build >/dev/null)
cargo build --manifest-path "$project_root/vision/Cargo.toml" --release \
  --bin train_board_policy --bin check_checkpoint_scope

set -o pipefail
node "$project_root/tools/render-vision-boards.mjs" \
  --curriculum "$curriculum" \
  --variants-per-failure 4 \
  --generate-boards 852 \
  --seed 2026090601 \
  2> "$artifact_root/render.log" \
| "$trainer" \
  --rendered-boards-stdin \
  --input-checkpoint "$checkpoint" \
  --output-root "$artifact_root" \
  --steps 1000 \
  --data-seed 590013 \
  --learning-rate 2e-5 \
  --dense-triple-loss \
  --shape-fill-decision-only \
| tee "$artifact_root/train.log"

for step in 250 1000; do
  candidate="$artifact_root/step-$step/set-vision.ftsv"
  "$scope_checker" "$checkpoint" "$candidate" \
    --allow-prefix attribute_head_2. \
    --allow-prefix attribute_head_3. \
    --allow-prefix fill_texture_head. \
    --allow-prefix attribute_relation_ \
    --allow-prefix set_relation_ \
    | tee "$artifact_root/step-$step/scope.log"
  shasum -a 256 "$candidate"
done
