#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
output_dir=${1:-"$project_root/vision-artifacts/atlases"}
seed_count=${2:-8}
seed_start=${3:-1}
theme_filter=${4:-all}
card_widths_arg=${5:-190}
chrome_bin='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
preview_log=$(mktemp /tmp/set-vision-preview.XXXXXX)

case "$theme_filter" in
  all) themes=(classic nocturne atelier signal) ;;
  classic|nocturne|atelier|signal) themes=("$theme_filter") ;;
  *)
    echo 'Theme must be all, classic, nocturne, atelier, or signal' >&2
    exit 2
    ;;
esac

IFS=',' read -r -a card_widths <<<"$card_widths_arg"
for card_width in "${card_widths[@]}"; do
  if [[ ! "$card_width" =~ ^[0-9]+$ ]] || ((card_width < 96 || card_width > 380)); then
    echo 'Card widths must be comma-separated integers from 96 through 380' >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
cd "$project_root"
npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort >"$preview_log" 2>&1 &
preview_pid=$!
trap 'kill "$preview_pid" 2>/dev/null || true' EXIT

for _attempt in {1..50}; do
  if ! kill -0 "$preview_pid" 2>/dev/null; then
    cat "$preview_log" >&2
    exit 1
  fi
  if curl --fail --silent http://127.0.0.1:4173/ >/dev/null; then
    break
  fi
  sleep 0.1
done

render_atlas() {
  local theme=$1
  local seed=$2
  local padded_seed
  local profile
  local output_name
  printf -v padded_seed '%04d' "$seed"
  output_name="$theme-$padded_seed"
  if ((card_width != 190)); then
    output_name+="-w$card_width"
  fi
  profile=$(mktemp -d /tmp/set-vision-chrome.XXXXXX)
  local status=0
  timeout --kill-after=1 8 "$chrome_bin" \
    --headless=new \
    --disable-gpu \
    --hide-scrollbars \
    --force-device-scale-factor=1 \
    --window-size="$window_width,$window_height" \
    --user-data-dir="$profile" \
    --screenshot="$output_dir/$output_name.png" \
    "http://127.0.0.1:4173/?vision-atlas=$theme&vision-seed=$seed&vision-card-width=$card_width" \
    >/dev/null 2>/dev/null || status=$?
  if ((status != 0 && status != 124 && status != 137)); then
    return "$status"
  fi
  test -s "$output_dir/$output_name.png"
}

seed_end=$((seed_start + seed_count - 1))
pids=()
pending=0
wait_for_batch() {
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  pids=()
  pending=0
}

for card_width in "${card_widths[@]}"; do
  card_height=$(((card_width * 140 + 95) / 190))
  window_width=$((20 + 9 * card_width + 8 * 4))
  window_height=$((20 + 9 * card_height + 8 * 4))
  for seed in $(seq "$seed_start" "$seed_end"); do
    for theme in "${themes[@]}"; do
      render_atlas "$theme" "$seed" &
      pids+=("$!")
      pending=$((pending + 1))
      if ((pending >= 4)); then
        wait_for_batch
      fi
    done
  done
done
if ((pending > 0)); then
  wait_for_batch
fi

printf 'Rendered %s procedural atlases at card widths %s in %s\n' \
  "$((seed_count * ${#themes[@]} * ${#card_widths[@]}))" "$card_widths_arg" "$output_dir"
