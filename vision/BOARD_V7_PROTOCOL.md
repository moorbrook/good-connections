# Board-v7 adversarial raw-perception protocol

Board-v7 is a fresh capture-shift cycle for the pixel-only policy. It is
intentionally adversarial: the goal is to separate a representation that has
learned the card symbols from one that only works on the six capture profiles
used in board-v6. This is a raw-model experiment. No confidence guard,
abstention threshold, or repeated-frame filter may change the board result.
The research target is at least **99% raw board accuracy at 100% coverage**.

## Why this corpus is new

The board-v6 renderer already exercised clean, mild blur, 1.055x crop/zoom-in,
0.94x shadow/scale-down, low contrast, JPEG quality 52, four themes, unseen
seeds, DPR 1/2, and source widths from 101 through 377 CSS pixels. The audit
found five remaining blind spots:

- the blur profile is a mild isotropic 0.55px filter, not severe defocus;
- no capture profile rotates or tilts card contents;
- no profile partially occludes the symbols or stripe evidence;
- no profile tests a compressed colour gamut and hue shift; and
- no profile combines blur, crop, shadow, and contrast loss, while the smallest
  source width remains 96px.

Board-v7 adds one profile for each of those failure modes and a new 83px
source raster. The new widths and procedural seeds are disjoint from the
board-v4, board-v5, and board-v6 corpora. The old profiles remain in the
matrix so a result can be compared by profile rather than only by aggregate
accuracy.

## Capture profiles

The profile names are literal descriptions of the renderer intervention. All
transforms are applied inside each fixed card wrapper, so the atlas still
encodes a 9x9 grid and the Rust loader's card boundaries remain unchanged.

| Profile | Capture intervention |
| --- | --- |
| `clean` | No capture transform. |
| `blur` | Existing mild `blur(0.55px) contrast(0.92)`. |
| `defocus` | New severe `blur(1.15px) contrast(0.88)`. |
| `crop` | Existing `translate(3px,-2px) scale(1.055)` with clipped wrapper. |
| `shadow` | Existing 0.94x scale, offset, and 48% cast shadow. |
| `lowcontrast` | Existing contrast, brightness, and saturation reduction. |
| `jpeg` | Existing PNG → JPEG quality 52 → PNG round trip. |
| `tilt` | New clipped `rotate(2.2deg) scale(1.04)` card-content transform. |
| `occlusion` | New 46%-opacity 22px horizontal band across the centre of each card. |
| `gamut` | New saturation 0.45, hue rotation 24°, contrast 1.08, brightness 1.05. |
| `compound` | New clipped 1.065x crop plus 0.75px blur, contrast 0.82, saturation 0.72, and cast shadow. |

The occlusion band is deliberately translucent rather than a label overlay: a
human can still use the surviving outline and colour evidence, and the stripe
pattern remains visible above and below the band. It is a difficult but
human-solvable capture, not an information-erasure test. A model that relies on
a single uninterrupted stripe or symbol edge should reveal that dependency.
Profiles are fixed per atlas, not chosen after seeing a model prediction.

## Locked research corpus

Directory: `vision-artifacts/board-v7-research-atlases`

- seeds `15001`–`15003` (three independent procedural variations);
- all four themes: `classic`, `nocturne`, `atelier`, `signal`;
- true Chrome DPR 1, CSS widths `83`, `157`, and `293`, with all eleven
  profiles;
- true Chrome DPR 2, CSS widths `107` and `277`, with profiles `clean`,
  `crop`, `shadow`, `tilt`, and `occlusion`;
- 396 DPR-1 atlases plus 120 DPR-2 atlases = **516 PNG atlases**.

## Locked sealed corpus

Directory: `vision-artifacts/board-v7-sealed-atlases`

The sealed matrix is identical except for the procedural seeds, which are
`16001`–`16003`. It therefore contains **516 PNG atlases** with no source
seed overlap with research or any earlier board cycle. It may be rendered and
visually inspected before selection, but it must not be passed to the
evaluator while candidates are being ranked.

The sorted per-file aggregate SHA-256 digests are recorded in
[`BOARD_V7_CORPORA.sha256`](BOARD_V7_CORPORA.sha256) after file-count,
dimension, and visual checks complete. Once recorded, do not add, remove,
re-render, or rename a corpus file.

For a CSS width `w`, the expected PNG dimensions are
`(52 + 9w) * dpr` by `(52 + 9 round(140w/190)) * dpr`. The loader then area
resamples every card to its fixed 190x140 RGB input. The filename's `-dpr2`
token is part of the data contract; it supplies the physical padding and gap
factor to the loader.

## Raw evaluation and selection

Use the existing FrankenTorch evaluator without source changes:

```sh
cargo run --release --manifest-path vision/Cargo.toml -- \
  --eval-only \
  --eval-batches 1024 \
  --eval-boards 8192 \
  --validation-atlas-dir "$PWD/vision-artifacts/board-v7-research-atlases" \
  --output-dir <candidate-artifact-dir>
```

The evaluator's fixed triple RNG (`0xa11c_e5ed_5eed`) and board RNG
(`0xb0a4_d5e7_5eed`) remain the protocol seeds. Keep `--eval-batches 1024`
and `--eval-boards 8192` unchanged for every control and candidate. Report
aggregate and per-profile failures, plus theme, width, DPR, and failed
attribute counts. The board decision is always the highest-scoring raw
triple; every board counts, including low-margin decisions.

Rank candidates lexicographically by:

1. fewer raw board failures (primary);
2. higher raw triple accuracy (tie-breaker); and
3. fewer learned parameters (final tie-breaker).

Do not fit an abstention guard, inspect sealed predictions, or use cached
sealed examples to choose a topology. A candidate is eligible for the
one-time sealed evaluation only when its raw research accuracy is at least
99%. With 8,192 boards this means **at most 81 raw board failures** (82 would
be 98.999%); candidates below that threshold must leave the sealed corpus
evaluator-unopened. If multiple candidates clear the threshold, select the
research winner under the ranking above and evaluate that checkpoint once on a
separately fixed 10,000-board raw run; those results are an assessment, not a
tuning signal.

This protocol supports only finite-distribution claims such as “at least 99%
raw accuracy at full coverage in 8,192 research boards” or “0 failures in
10,000 sealed boards.” It is not a claim of universal 100% reliability, and the adversarial profiles are not
evidence that the browser can safely click when a human could not identify a
card.

## Reproduction and integrity checks

From the repository root, render the two DPR-1 and DPR-2 portions separately
because the DPR-2 matrix intentionally uses fewer profiles:

```sh
research="vision-artifacts/board-v7-research-atlases"
sealed="vision-artifacts/board-v7-sealed-atlases"
profiles='clean,blur,defocus,crop,shadow,lowcontrast,jpeg,tilt,occlusion,gamut,compound'
dpr2_profiles='clean,crop,shadow,tilt,occlusion'

tools/render-vision-atlases.sh "$research" 3 15001 all '83,157,293' "$profiles" 1
tools/render-vision-atlases.sh "$research" 3 15001 all '107,277' "$dpr2_profiles" 2
tools/render-vision-atlases.sh "$sealed" 3 16001 all '83,157,293' "$profiles" 1
tools/render-vision-atlases.sh "$sealed" 3 16001 all '107,277' "$dpr2_profiles" 2

node tools/verify-vision-corpus.mjs "$research" 516
node tools/verify-vision-corpus.mjs "$sealed" 516
tools/hash-vision-corpus.sh "$research"
tools/hash-vision-corpus.sh "$sealed"
```

Before recording the digests, inspect at least one full-deck atlas for every
theme, every new profile, both width extremes, both DPRs, and both seed ranges.
The full deck must show all 81 cards, intact grid boundaries, and the expected
capture transform; the sealed samples must not be fed into an evaluator during
this visual QA.
