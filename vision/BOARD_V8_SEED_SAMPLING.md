# Board-v8 seed-stratified training sampler

This is one training-data change for board-v8. It keeps the raw forced-action
objective, policy topology, pixel contract, and evaluator unchanged. It must be
ranked only on board-v8 research data; no v7 sealed predictions or v8 sealed
data are involved in this candidate.

## Hypothesis

Procedural renderer seeds change small but decision-relevant details such as
anti-aliasing, stripe phase, shape placement, and theme-specific rasterization.
The existing training sampler balances theme and source-raster bucket, then
samples a file from that bucket. When a bucket has an unequal number of
capture/width variants per seed, file sampling gives some procedural seeds more
training probability than others. Even when the counts happen to be equal,
independent per-example draws can reuse one seed several times in a batch.

Hypothesis: sampling procedural seed families uniformly and walking a
randomized seed order within each batch will reduce seed-specific memorization
and improve raw forced-action accuracy on unseen board-v8 renderer seeds, while
leaving the source-raster balance and label distribution unchanged.

This is a data-distribution hypothesis, not a scaling claim. The v7 protocol's
research matrix deliberately used disjoint procedural seeds (`15001`--`15003`)
and multiple widths, themes, DPRs, and capture profiles; this candidate targets
generalization to the next disjoint seed set rather than adding another
hand-tuned pixel transform.

## Implementation

`AtlasCollection::load` now builds a second, training-only index:

```text
theme -> source raster -> procedural seed -> capture/width variants
```

The existing flat index remains intact for `make_batch`, which is used by
probes and evaluation. Both native training loops call
`make_training_batch` instead. For each training batch, one RNG-derived offset
is chosen. Within each selected theme/raster bucket, sample `i` chooses seed
`(offset + i) mod seed_count`, then chooses one of that seed's available
capture/width variants uniformly. Thus a bucket with `k` seed families visits
all `k` families before repeating, with a randomized starting point. The
shared project `Rng` keeps MPSGraph and FrankenTorch batches identical for the
same data seed.

The first numeric filename token is the seed key. Names without a numeric token
fall into an `unknown` family and retain valid sampling behavior. No pixels are
changed by this candidate, and `make_board`/evaluation still use
`append_card(..., false)` through the unchanged flat sampling path.

The only runtime cost is one 64-bit RNG draw per training batch plus two index
lookups per training example; atlas pixel storage and model compute are
unchanged. The seed index adds only `O(number of atlas files)` integer entries.

## Focused checks

The data crate tests cover:

- numeric seed extraction while ignoring profile/DPR/width suffixes;
- randomized seed-order traversal without repeating a seed until the pool is
  exhausted; and
- the existing augmentation determinism, bounds, and `augment=false` exact-copy
  contracts.

Run the checks with:

```sh
cargo fmt --manifest-path vision-data/Cargo.toml -- --check
cargo clippy --manifest-path vision-data/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path vision-data/Cargo.toml
cargo check --manifest-path vision/Cargo.toml
cargo check --manifest-path vision-mpsgraph/Cargo.toml
```

## Reproduction command

Use the same optimizer, model initialization, augmentation setting, atlas
directory, and step budget as the board-v8 baseline; only the training sampler
differs. The MPSGraph training command is:

```sh
cargo run --release --manifest-path vision-mpsgraph/Cargo.toml -- \
  --steps 20000 \
  --data-seed 490009 \
  --init-seed 3935559000370003845 \
  --output vision-artifacts/board-v8-seed-stratified/set-vision.ftsv
```

The default augmentation setting is intentionally retained so the experiment
isolates seed sampling from disabling the existing raw-model training recipe.
Evaluate the resulting checkpoint only against the locked board-v8 research
corpus under its raw, 100%-coverage protocol. Do not pass it to any sealed
evaluator unless it first clears that protocol's research gate.

Status: implementation candidate only; no board-v8 training or evaluation was
run in this change.
