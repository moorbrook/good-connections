#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! -d "$1" ]]; then
  echo 'Usage: tools/hash-vision-corpus.sh DIRECTORY' >&2
  exit 2
fi

directory=$(cd "$1" && pwd)
(
  cd "$directory"
  find . -maxdepth 1 -type f -name '*.png' -print0 \
    | sort -z \
    | xargs -0 shasum -a 256
) | shasum -a 256 | awk '{ print $1 }'
