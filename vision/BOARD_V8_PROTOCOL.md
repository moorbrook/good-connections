# Board-v8 adversarial raw-perception protocol

Board-v8 is a new procedural capture cycle for the pixel-only policy. It is
locked for a **99% raw board-accuracy goal at 100% coverage**. The result is a
finite-distribution measurement, not a universal reliability claim. No
confidence guard, abstention threshold, repeated-frame filter, or symbolic
solver may change a board result.

This lock was made before any v8 model evaluation. The research and sealed
corpora are complete, visually inspected, and recorded in
`BOARD_V8_CORPORA.sha256`. The sealed corpus must remain evaluator-unopened
until a research winner passes the raw gate.

## What is new

The v8 cycle keeps the v7 adversarial profile matrix so profile-level changes
remain comparable, but moves to new procedural seeds and new source raster
widths. The v8 widths are absent from the inspected board-v5, board-v6, and
board-v7 corpus filenames, including board-v7 training, research, and sealed
matrices. Research and sealed use identical capture geometry but disjoint seed
ranges.

The five adversarial profiles remain intentionally human-solvable:

- `defocus`: severe but finite blur;
- `tilt`: a clipped, small card-content rotation;
- `occlusion`: a translucent centre band that leaves outline, colour, and
  stripe evidence above and below it;
- `gamut`: a compressed and hue-shifted colour presentation; and
- `compound`: a finite combination of crop, blur, contrast, saturation, and
  shadow.

The occlusion band is not an information-erasure test. A human can still use
the surviving card evidence, while a model that depends on one uninterrupted
stripe or symbol edge is exposed.

## Capture profiles

The profile names are literal renderer interventions. All transforms are
applied inside each fixed card wrapper, preserving the 9x9 atlas grid and the
Rust loader's card boundaries.

| Profile | Capture intervention |
| --- | --- |
| `clean` | No capture transform. |
| `blur` | `blur(0.55px) contrast(0.92)`. |
| `defocus` | `blur(1.15px) contrast(0.88)`. |
| `crop` | `translate(3px,-2px) scale(1.055)` with clipping. |
| `shadow` | 0.94x scale, offset, and a 48% cast shadow. |
| `lowcontrast` | Contrast, brightness, and saturation reduction. |
| `jpeg` | PNG → JPEG quality 52 → PNG round trip. |
| `tilt` | Clipped `rotate(2.2deg) scale(1.04)` card-content transform. |
| `occlusion` | A 46%-opacity, 22px horizontal band across the card centre. |
| `gamut` | Saturation 0.45, hue rotation 24°, contrast 1.08, brightness 1.05. |
| `compound` | Clipped 1.065x crop, 0.75px blur, contrast 0.82, saturation 0.72, and cast shadow. |

## Locked research corpus

Directory: `vision-artifacts/board-v8-research-atlases`

- Procedural seeds: `18001`, `18002`, `18003`.
- Themes: `classic`, `nocturne`, `atelier`, `signal`.
- DPR 1, CSS widths `79`, `167`, `323`, with all eleven profiles.
- DPR 2, CSS widths `115`, `281`, with `clean`, `crop`, `shadow`, `tilt`,
  and `occlusion`.
- 396 DPR-1 atlases plus 120 DPR-2 atlases = **516 PNG atlases**.
- Each seed contributes 172 atlases.
- Sorted per-file aggregate SHA-256:
  `4c5fcba428b7a48be8b1ab69ea6c9ac72395b2b59d1827d06fe3025cee94e62d`.

## Locked sealed corpus

Directory: `vision-artifacts/board-v8-sealed-atlases`

The sealed matrix is identical to research except for the procedural seeds:
`19001`, `19002`, and `19003`. It contains **516 PNG atlases**, 172 per
seed, and has no seed overlap with the v8 research corpus or any inspected
earlier board cycle. Its sorted per-file aggregate SHA-256 is
`9dfbbc84fc8e97c65f9070e43528be11c66073fa9e4b360ffabedd9077bfbe67`.

After this lock, do not add, remove, re-render, or rename a v8 sealed file.
Do not pass the sealed directory to an evaluator while candidates are being
ranked. The sealed digest is an integrity record, not a prediction result.

For a CSS width `w`, expected PNG dimensions are
`(52 + 9w) * dpr` by `(52 + 9 round(140w/190)) * dpr`. Those are source-capture
dimensions; the loader area-resamples each card to the fixed 30x21 RGB model
input. The filename's `-dpr2` token is part of the data contract.

## Locked training corpus

Directory: `vision-artifacts/board-v8-training-atlases`

- Procedural seeds: `20001`, `20002`, `20003`, `20004`.
- Themes: `classic`, `nocturne`, `atelier`, `signal`.
- DPR 1, CSS widths `89`, `193`, `347`, with all eleven profiles.
- DPR 2, CSS widths `127`, `263`, with `clean`, `crop`, `shadow`, `tilt`,
  and `occlusion`.
- 528 DPR-1 atlases plus 160 DPR-2 atlases = **688 PNG atlases**.
- Each seed and theme contributes 172 atlases.
- Sorted per-file aggregate SHA-256:
  `a8b5c19760547bc262a0946b2f15b31c15c39a5dc5feeb42124d69d61e5437e4`.

The seeds and widths are disjoint from the v8 research and sealed corpora and
from the inspected v7 training, research, and sealed matrices. Native QA of
the `nocturne` DPR-2 occlusion and `signal` compound samples confirmed intact
9x9 grids and readable attribute evidence. This digest was locked before v8
training began and is also recorded in `BOARD_V8_TRAINING.sha256`.

## Rendering and integrity

From the repository root, render the portions separately. The renderer uses
the production card component and the production Chrome capture path:

```sh
research="vision-artifacts/board-v8-research-atlases"
sealed="vision-artifacts/board-v8-sealed-atlases"
profiles='clean,blur,defocus,crop,shadow,lowcontrast,jpeg,tilt,occlusion,gamut,compound'
dpr2_profiles='clean,crop,shadow,tilt,occlusion'

tools/render-vision-atlases.sh "$research" 3 18001 all '79,167,323' "$profiles" 1
tools/render-vision-atlases.sh "$research" 3 18001 all '115,281' "$dpr2_profiles" 2
tools/render-vision-atlases.sh "$sealed" 3 19001 all '79,167,323' "$profiles" 1
tools/render-vision-atlases.sh "$sealed" 3 19001 all '115,281' "$dpr2_profiles" 2

node tools/verify-vision-corpus.mjs "$research" 516
node tools/verify-vision-corpus.mjs "$sealed" 516
tools/hash-vision-corpus.sh "$research"
tools/hash-vision-corpus.sh "$sealed"
```

The locked validation counts are 129 files per theme, 172 per seed, 396
DPR-1 files, 120 DPR-2 files, and 516 files per corpus. Before recording a
digest, inspect full-deck samples for every theme, every new profile, both
width extremes, both DPRs, and both seed ranges. The locked QA samples were:

```text
research:
  classic-18001-w79.png
  nocturne-18002-pcompound-w323.png
  atelier-18003-pocclusion-dpr2-w115.png
  signal-18001-pgamut-w323.png
  classic-18002-pdefocus-w167.png
  signal-18003-ptilt-dpr2-w281.png

sealed:
  classic-19001-w79.png
  nocturne-19002-pcompound-w323.png
  atelier-19003-pocclusion-dpr2-w115.png
  signal-19001-pgamut-w323.png
  classic-19002-pdefocus-w167.png
  signal-19003-ptilt-dpr2-w281.png
```

Native inspection confirmed all 9x9 grids and card boundaries were intact.
The occlusion samples retained evidence on both sides of the band; defocus,
tilt, gamut, and compound samples showed the intended finite capture change
without turning the cards into missing-information examples.

## Raw research selection

The full research corpus is evaluated as three sequential 172-atlas shards to
give each procedural seed an exact independently seeded share of the score.
The 30x21 model input does not require this split for memory; the shard layout
is a balanced, locked evaluation protocol. Every board counts at 100% raw
coverage. The fixed research allocation is:

| Atlas seed | Triple RNG seed | Board RNG seed | Triple batches | Raw boards |
| ---: | ---: | ---: | ---: | ---: |
| 18001 | 287145488695021 | 294222009835245 | 342 | 2,731 |
| 18002 | 287145488695022 | 294222009835246 | 341 | 2,731 |
| 18003 | 287145488695023 | 294222009835247 | 341 | 2,730 |
| **Total** |  |  | **1,024** | **8,192** |

The allocation produces exactly 65,536 triple examples and 8,192 raw boards.
The shard processes run sequentially. Record the per-shard and aggregate raw
board failures, triple accuracy, attribute accuracy, profile failures, theme
failures, width failures, DPR failures, and failed attribute counts.

Four independently trained v8 replicas are predeclared before selection. The
replica outputs are combined with fixed uniform weights `[0.25, 0.25, 0.25,
0.25]`. Do not tune weights, fit a guard, or search ensemble coefficients on
the research corpus. The four-replica aggregate is the raw candidate result.
Rank eligible candidates by fewer raw board failures, then higher raw triple
accuracy, then fewer learned parameters.

The research gate is at least 99% raw board accuracy at full coverage: at most
81 failures in 8,192 boards. A candidate with 82 or more failures is not
eligible and must leave the sealed corpus evaluator-unopened.

## One-time sealed assessment

Only a research winner that clears the raw gate may receive one sealed run.
The sealed run is a separate fixed 10,000-board raw assessment, not a tuning
signal. Use three sequential shards with 3,334 / 3,333 / 3,333 boards for
seeds 19001 / 19002 / 19003 and fixed triple and board RNG seeds:

| Atlas seed | Triple RNG seed | Board RNG seed | Triple batches | Raw boards |
| ---: | ---: | ---: | ---: | ---: |
| 19001 | 317145488695021 | 304222009835245 | 342 | 3,334 |
| 19002 | 317145488695022 | 304222009835246 | 341 | 3,333 |
| 19003 | 317145488695023 | 304222009835247 | 341 | 3,333 |
| **Total** |  |  | **1,024** | **10,000** |

The sealed result must report only the predeclared raw outcome and must not
change the model, weights, or protocol afterward. This protocol supports only
finite-distribution claims such as “at least 99% raw accuracy at full coverage
in 8,192 research boards” or a stated raw result in 10,000 sealed boards.
