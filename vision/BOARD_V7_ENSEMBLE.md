# Board-v7 raw multi-checkpoint ensemble evaluator

The native `vision` evaluator has optional, evaluation-only two-, three-, and
four-checkpoint modes for measuring whether independently trained instances
of the existing specialist-dilated-residual policy complement one another. It
does not change the policy topology, browser checkpoint, or default
single-checkpoint path.

Run it with two checkpoint directories:

```sh
cargo run --release --manifest-path vision/Cargo.toml -- \
  --eval-only \
  --validation-atlas-dir "$RESEARCH_SHARD_DIR" \
  --output-dir vision-artifacts/candidate-a \
  --ensemble-output-dir vision-artifacts/candidate-b \
  --ensemble-primary-weight "$PRIMARY_WEIGHT" \
  --eval-batches "$SHARD_BATCHES" --eval-boards "$SHARD_BOARDS" \
  --triple-seed 177145488695021 --board-seed 194222009835245
```

Use one board-v7 research shard per process, with the allocation in
`BOARD_V7_PROTOCOL.md`; do not load all 516 atlases into one process. The
primary weight must be finite and strictly between zero and one; it defaults to
`0.5` when an ensemble is requested and is accepted only with both
`--ensemble-output-dir` and `--eval-only`. For a non-default weight, the
evaluator computes `W * primary + (1 - W) * secondary` for every set logit,
board score, and attribute logit. The `0.5` path retains the existing midpoint
operation exactly.

To evaluate three checkpoints, add a third directory and three positive
weights. The evaluator normalizes the supplied weights to sum to one:

```sh
  --ensemble-third-output-dir vision-artifacts/candidate-c \
  --ensemble-weights 2 3 5
```

The three-model path computes `W1 * primary + W2 * secondary + W3 * third` for
every set logit, board score, and attribute logit. `--ensemble-weights` is
required with the third checkpoint and cannot be combined with the two-model
`--ensemble-primary-weight` flag. A third checkpoint without the second, or
weights supplied without a third checkpoint, is rejected.

To evaluate four checkpoints, add a fourth directory and four positive
weights. The evaluator normalizes these weights to sum to one:

```sh
  --ensemble-third-output-dir vision-artifacts/candidate-c \
  --ensemble-fourth-output-dir vision-artifacts/candidate-d \
  --ensemble-four-weights 1 2 3 4
```

The four-model path computes `W1 * primary + W2 * secondary + W3 * third +
W4 * fourth` for every set logit, board score, and attribute logit.
`--ensemble-four-weights` requires both the second and third checkpoint
directories, cannot be combined with `--ensemble-weights` or
`--ensemble-primary-weight`, and a fourth checkpoint without four weights is
rejected. The three-checkpoint invocation remains unchanged when the fourth
directory is omitted.

Each checkpoint is loaded into its own strict FrankenTorch session. For the
triple metrics, the evaluator averages corresponding set logits and attribute
logits. For board metrics, it averages corresponding learned triple scores and
per-card attribute logits, then chooses the highest-scoring triple at full raw
coverage. The ensemble path performs no confidence guard, abstention, symbol
solver, voting tie-break, or sealed-corpus access. It rejects missing or
incompatible output, mismatched triple ordering or logit lengths, and
non-finite logits.

The ensemble flags are rejected during training and probe-only runs. An
invocation without `--ensemble-output-dir` retains the existing single-model
evaluator and its default behavior; an invocation with only the second
checkpoint retains the existing two-model behavior.

This branch adds evaluator and parser tests only; it intentionally does not
run board-v7 research or sealed evaluation. Record any future measurement in
the board-v7 research log with all checkpoint hashes, normalized weights, and
the exact raw evaluation allocation.
