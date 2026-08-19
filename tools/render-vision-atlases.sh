#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
output_dir=${1:-"$project_root/vision-artifacts/atlases"}
seed_count=${2:-8}
seed_start=${3:-1}
theme_filter=${4:-all}
card_widths_arg=${5:-190}
capture_profiles_arg=${6:-clean}
device_scales_arg=${7:-1}
chrome_bin='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
preview_log=$(mktemp /tmp/set-vision-preview.XXXXXX)
preview_port=${VISION_PREVIEW_PORT:-4173}
if [[ ! "$preview_port" =~ ^[0-9]+$ ]] || ((preview_port < 1024 || preview_port > 65535)); then
  echo 'VISION_PREVIEW_PORT must be an integer from 1024 through 65535' >&2
  exit 2
fi

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
  if [[ ! "$card_width" =~ ^[0-9]+$ ]] || ((card_width < 72 || card_width > 380)); then
    echo 'Card widths must be comma-separated integers from 72 through 380' >&2
    exit 2
  fi
done

IFS=',' read -r -a capture_profiles <<<"$capture_profiles_arg"
for capture_profile in "${capture_profiles[@]}"; do
  case "$capture_profile" in
    clean|blur|crop|shadow|lowcontrast|jpeg|defocus|tilt|occlusion|gamut|compound) ;;
    *)
      echo 'Capture profiles must be comma-separated clean, blur, crop, shadow, lowcontrast, jpeg, defocus, tilt, occlusion, gamut, or compound' >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a device_scales <<<"$device_scales_arg"
for device_scale in "${device_scales[@]}"; do
  if [[ ! "$device_scale" =~ ^[12]$ ]]; then
    echo 'Device scales must be comma-separated 1 or 2' >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
cd "$project_root"
npm run build
npm run preview -- --host 127.0.0.1 --port "$preview_port" --strictPort >"$preview_log" 2>&1 &
preview_pid=$!
trap 'kill "$preview_pid" 2>/dev/null || true' EXIT

for _attempt in {1..50}; do
  if ! kill -0 "$preview_pid" 2>/dev/null; then
    cat "$preview_log" >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:$preview_port/" >/dev/null; then
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
  if [[ "$capture_profile" != clean ]]; then
    output_name+="-p$capture_profile"
  fi
  if ((device_scale != 1)); then
    output_name+="-dpr$device_scale"
  fi
  if ((card_width != 190)); then
    output_name+="-w$card_width"
  fi
  if [[ "$capture_profile" != jpeg && -s "$output_dir/$output_name.png" ]]; then
    return 0
  fi
  profile=$(mktemp -d /tmp/set-vision-chrome.XXXXXX)
  local status=0
  timeout --kill-after=1 8 "$chrome_bin" \
    --headless=new \
    --disable-gpu \
    --hide-scrollbars \
    --force-device-scale-factor="$device_scale" \
    --window-size="$window_width,$window_height" \
    --user-data-dir="$profile" \
    --screenshot="$output_dir/$output_name.png" \
    "http://127.0.0.1:$preview_port/?vision-atlas=$theme&vision-seed=$seed&vision-card-width=$card_width&vision-capture=$capture_profile" \
    >/dev/null 2>/dev/null || status=$?
  if ((status != 0 && status != 124 && status != 137)); then
    return "$status"
  fi
  test -s "$output_dir/$output_name.png"
  if [[ "$capture_profile" == jpeg ]]; then
    local conversion_dir
    local jpeg_file
    local decoded_file
    conversion_dir=$(mktemp -d /tmp/set-vision-convert.XXXXXX)
    jpeg_file="$conversion_dir/frame.jpg"
    decoded_file="$conversion_dir/frame.png"
    sips -s format jpeg -s formatOptions 52 "$output_dir/$output_name.png" --out "$jpeg_file" >/dev/null
    sips -s format png "$jpeg_file" --out "$decoded_file" >/dev/null
    mv "$decoded_file" "$output_dir/$output_name.png"
    trash "$conversion_dir"
  fi
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

for device_scale in "${device_scales[@]}"; do
  for capture_profile in "${capture_profiles[@]}"; do
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
  done
done
if ((pending > 0)); then
  wait_for_batch
fi

printf 'Rendered %s procedural atlases at card widths %s, profiles %s, DPRs %s in %s\n' \
  "$((seed_count * ${#themes[@]} * ${#card_widths[@]} * ${#capture_profiles[@]} * ${#device_scales[@]}))" \
  "$card_widths_arg" "$capture_profiles_arg" "$device_scales_arg" "$output_dir"
