#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
model_source="$project_root/vision-artifacts/model/set-vision.ftsv"
wasm_target="$project_root/vision-wasm/target/wasm32-unknown-unknown/release/set_vision_wasm.wasm"

test -s "$model_source"
command -v wasm-bindgen >/dev/null || {
  echo 'wasm-bindgen-cli 0.2.127 is required (cargo install wasm-bindgen-cli --version 0.2.127 --locked)' >&2
  exit 1
}
cargo build \
  --manifest-path "$project_root/vision-wasm/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --release
wasm-bindgen \
  --target web \
  --no-typescript \
  --out-dir "$project_root/extension" \
  --out-name set_vision_runtime \
  "$wasm_target"
printf 'Extension ready at %s\n' "$project_root/extension"
