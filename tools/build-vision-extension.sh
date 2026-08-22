#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
model_source="$project_root/vision-artifacts/model/set-vision.ftsv"
wasm_target="$project_root/vision-wasm/target/wasm32-unknown-unknown/release/set_vision_wasm.wasm"
enabled_features=()
case "${VISION_HIGH_RESOLUTION:-0}" in
  0|'') ;;
  1) enabled_features+=(high-resolution) ;;
  *) echo 'VISION_HIGH_RESOLUTION must be 0 or 1' >&2; exit 2 ;;
esac
case "${VISION_ENSEMBLE_FOUR:-0}" in
  0|'') ;;
  1) enabled_features+=(ensemble-four) ;;
  *) echo 'VISION_ENSEMBLE_FOUR must be 0 or 1' >&2; exit 2 ;;
esac
cargo_args=(
  build
  --manifest-path "$project_root/vision-wasm/Cargo.toml"
  --target wasm32-unknown-unknown
)
if ((${#enabled_features[@]} > 0)); then
  feature_list=$(IFS=,; echo "${enabled_features[*]}")
  cargo_args+=(--features "$feature_list")
fi
cargo_args+=(--release)

test -s "$model_source"
command -v wasm-bindgen >/dev/null || {
  echo 'wasm-bindgen-cli 0.2.127 is required (cargo install wasm-bindgen-cli --version 0.2.127 --locked)' >&2
  exit 1
}
cargo "${cargo_args[@]}"
wasm-bindgen \
  --target web \
  --no-typescript \
  --out-dir "$project_root/extension" \
  --out-name set_vision_runtime \
  "$wasm_target"
printf 'Extension ready at %s\n' "$project_root/extension"
