#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
artifact_root="$project_root/vision-artifacts/active-v29"
training_atlases="$artifact_root/training-atlases"
eligibility_atlases="$artifact_root/eligibility-atlases"
profiles=clean,blur,crop,shadow,lowcontrast,jpeg,defocus,tilt,occlusion,gamut,compound

mkdir -p "$artifact_root"

VISION_RENDER_JOBS=8 "$project_root/tools/render-vision-atlases.sh" \
  "$training_atlases" 4 29001 all 96,144,208,320 "$profiles" 1,2
node "$project_root/tools/verify-vision-corpus.mjs" "$training_atlases" 1408 \
  | tee "$artifact_root/training-corpus.json"
"$project_root/tools/hash-vision-corpus.sh" "$training_atlases" \
  | tee "$artifact_root/training-corpus.sha256"

VISION_RENDER_JOBS=8 "$project_root/tools/render-vision-atlases.sh" \
  "$eligibility_atlases" 2 29101 all 104,152,216,344 "$profiles" 1,2
node "$project_root/tools/verify-vision-corpus.mjs" "$eligibility_atlases" 704 \
  | tee "$artifact_root/eligibility-corpus.json"
"$project_root/tools/hash-vision-corpus.sh" "$eligibility_atlases" \
  | tee "$artifact_root/eligibility-corpus.sha256"
