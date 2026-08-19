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

## Locked training corpus

Directory: `vision-artifacts/board-v7-training-atlases`

This corpus is training data only. It is never a validation input, must not be
passed as `--validation-atlas-dir`, and has no evaluator result. It targets the
current research diagnosis (shadow, crop, low-contrast, and gamut failures,
especially in the nocturne theme) while keeping source seeds disjoint from the
research and sealed corpora:

- seeds `17001`–`17002`, all four themes, DPR 1 widths `97`, `211`, and `349`,
  with all eleven profiles: 264 PNG atlases;
- the same seeds and themes at true Chrome DPR 2, widths `139` and `251`,
  with `clean`, `crop`, `shadow`, `tilt`, and `occlusion`: 80 PNG atlases;
- **344 PNG atlases total**, 172 per seed.

The chosen widths are absent from the board-v7 research/sealed matrices and
from the board-v4/v5/v6 research matrices. The sorted per-file aggregate
digest is locked in [`BOARD_V7_TRAINING.sha256`](BOARD_V7_TRAINING.sha256).
Use this directory only as the training `--atlas-dir`; do not copy it into a
validation or sealed artifact directory.

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

The full corpus is intentionally evaluated as three sequential 172-atlas
shards. Loading all 516 atlases in one process exceeds the 24 GB memory budget
(`516 * 81 * 190 * 140 * 3 * f64` is about 26.7 GB before allocator overhead).
An eval-only process skips the training atlas and holds only its validation
shard. Use the deterministic research helper, which verifies the locked corpus
digest, builds the evaluator, and writes one log per shard:

```sh
tools/evaluate-board-v7-research-shards.sh \
  <candidate-checkpoint-dir> <new-output-dir>
```

The helper's fixed allocation is:

| atlas seed | triple seed | board seed | triple batches | raw boards |
| ---: | ---: | ---: | ---: | ---: |
| 15001 | 177145488695021 | 194222009835245 | 342 | 2,731 |
| 15002 | 177145488695022 | 194222009835246 | 341 | 2,731 |
| 15003 | 177145488695023 | 194222009835247 | 341 | 2,730 |
| **total** |  |  | **1,024** | **8,192** |

The distinct shard RNGs are the default protocol seeds
(`0xa11c_e5ed_5eed` and `0xb0a4_d5e7_5eed`) plus the shard index. The
evaluator also accepts these seeds explicitly as `--triple-seed N` and
`--board-seed N`; the defaults remain for compatibility, but a v7 result must
record the per-shard values above. Run shards sequentially, never in parallel.
The helper's `results.tsv` and `aggregate.txt` must show exactly 65,536
triples and 8,192 raw boards. Report aggregate and per-profile failures, plus
theme, width, DPR, and failed attribute counts. The board decision is always
the highest-scoring raw triple; every board counts, including low-margin
decisions.

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
tuning signal. The sealed run must also be sequential and memory-bounded, with
the fixed sealed-board allocation 3,334 / 3,333 / 3,333 for seeds 16001 /
16002 / 16003, respectively, and must aggregate exactly 10,000 boards. Do
not open the sealed corpus or create sealed predictions before a candidate has
cleared the research gate.

This protocol supports only finite-distribution claims such as “at least 99%
raw accuracy at full coverage in 8,192 research boards” or “0 failures in
10,000 sealed boards.” It is not a claim of universal 100% reliability, and the adversarial profiles are not
evidence that the browser can safely click when a human could not identify a
card.

## Reproduction and integrity checks

From the repository root, render the two DPR-1 and DPR-2 portions separately
because the DPR-2 matrix intentionally uses fewer profiles:

```sh
training="vision-artifacts/board-v7-training-atlases"
research="vision-artifacts/board-v7-research-atlases"
sealed="vision-artifacts/board-v7-sealed-atlases"
profiles='clean,blur,defocus,crop,shadow,lowcontrast,jpeg,tilt,occlusion,gamut,compound'
dpr2_profiles='clean,crop,shadow,tilt,occlusion'

tools/render-vision-atlases.sh "$training" 2 17001 all '97,211,349' "$profiles" 1
tools/render-vision-atlases.sh "$training" 2 17001 all '139,251' "$dpr2_profiles" 2
tools/render-vision-atlases.sh "$research" 3 15001 all '83,157,293' "$profiles" 1
tools/render-vision-atlases.sh "$research" 3 15001 all '107,277' "$dpr2_profiles" 2
tools/render-vision-atlases.sh "$sealed" 3 16001 all '83,157,293' "$profiles" 1
tools/render-vision-atlases.sh "$sealed" 3 16001 all '107,277' "$dpr2_profiles" 2

node tools/verify-vision-corpus.mjs "$training" 344
node tools/verify-vision-corpus.mjs "$research" 516
node tools/verify-vision-corpus.mjs "$sealed" 516
tools/hash-vision-corpus.sh "$training"
tools/hash-vision-corpus.sh "$research"
tools/hash-vision-corpus.sh "$sealed"
```

Before recording the digests, inspect at least one full-deck atlas for every
theme, every new profile, both width extremes, both DPRs, and both seed ranges.
The full deck must show all 81 cards, intact grid boundaries, and the expected
capture transform; the sealed samples must not be fed into an evaluator during
this visual QA.
