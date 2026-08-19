# Board-v7 raw-model augmentation candidate

This branch adds a deterministic, on-the-fly augmentation candidate to
`vision-data`. It is training-only: `AtlasCollection::make_board` and the raw
evaluator still use unmodified cards, and no board-v7 sealed corpus was read.
The candidate does not change the policy topology, action contract, evaluator,
or inference preprocessing.

## Sampling contract

Each card is sampled independently from the existing project `Rng`. Both the
MPSGraph and FrankenTorch trainers call the same `set-vision-data` code, so a
shared seed produces the same transformed batch. Sampling order is fixed in
`AugmentationPlan::sample`; changing it changes the candidate's data stream.

The transform plan is bounded as follows:

| Transform | Probability | Range / behavior | Label-safety bound |
| --- | ---: | --- | --- |
| Scale | 0.38 | isotropic factor `0.93..=1.07` | centered, clamped bilinear resample |
| Crop / translation | 0.32 independently per axis | x `-1.2..=1.2` px; y `-0.8..=0.8` px | output remains the same `30x21` card crop |
| Tilt | 0.28 | rotation `-4..=4` degrees | centered, clamped bilinear resample |
| Blur | 0.20 | 3x3 `[1,2,1] x [1,2,1] / 16` blend `0.35..=0.75` | no spatial label change |
| Directional shadow | 0.18 | linear edge shading, strength `0.08..=0.24`, random angle `0..2pi` | multiplier remains at least `0.76` |
| Sensor gamut | 0.35 | per-channel gain `0.97..=1.03`, offset `-0.008..=0.008` | small neutral gamut drift; no hue remapping |
| Horizontal-band occlusion | 0.16 | full-width band, height `2..=8` input px; top `0.35..=0.56` of the 21px card | alpha `0.30..=0.50`, dark tint `0.04..=0.16`; fractional edge coverage preserves evidence above and below |
| Existing photometric noise | always when `augment=true` | contrast `0.86..=1.14`; brightness `-0.055..=0.055`; per-pixel noise `-0.018..=0.018` | final pixels clamped to `[0,1]` |

The execution order is geometry, blur, existing photometric transform, gamut,
shadow, horizontal-band occlusion, and final clamp. The small geometric ranges
and central, translucent occlusion are intentional: they model the board-v7
capture band without destroying the symbol, count, fill, or colour evidence
needed by the labels. The band height is deliberately randomized in input
pixels; in the renderer's 190x140 reference-card coordinates, the board-v7
22px CSS band corresponds to roughly 14--50 pixels across the locked source
widths.

## Focused verification

`vision-data` now has tests for:

- exact deterministic output for equal seeds;
- output length, finiteness, and `[0,1]` bounds over 512 sampled cards;
- geometry and horizontal-band bounds over 10,000 sampled plans, including
  fractional edge coverage; and
- the existing `augment=false` exact-copy contract.

Commands:

```text
cargo fmt --manifest-path vision-data/Cargo.toml -- --check
cargo clippy --manifest-path vision-data/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path vision-data/Cargo.toml
```

No full training or evaluation run is part of this candidate. The ignored
release microbenchmark can be run with:

```text
cargo test --release --manifest-path vision-data/Cargo.toml \
  augmentation_microbenchmark -- --ignored --nocapture
```

On the Apple Silicon development machine, the horizontal-band variant measured
4,096 synthetic `30x21x3` cards in 39.452167 ms, or 9,631.9 ns/card. The
previous edge-occlusion candidate measured 8,430.4 ns/card in the preceding
run, so replacing the edge rectangle with fractional full-width bands added
about 1.20 us/card (roughly 14% in these runs). The previous
contrast/brightness/noise-only implementation in the same benchmark took
16.172209 ms, or 3,948.3 ns/card, making the new candidate about 2.44x that
legacy augmentation path. The copy-only baseline was 1,268.6 ns/card. These
are data-pipeline costs, not end-to-end training throughput, and should be
remeasured alongside a training run if the candidate is selected.

## Status

Candidate only. It has not been trained, ranked against board-v7 research
boards, or promoted to deployment. Commit hash and exact source changes should
be recorded by the experiment owner after review.
