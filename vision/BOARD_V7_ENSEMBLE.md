# Board-v7 raw two-checkpoint ensemble evaluator

The native `vision` evaluator has an optional, evaluation-only two-checkpoint
mode for measuring whether two independently trained instances of the existing
specialist-dilated-residual policy complement one another. It does not change
the policy topology, browser checkpoint, or default single-checkpoint path.

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

Each checkpoint is loaded into its own strict FrankenTorch session. For the
triple metrics, the evaluator averages corresponding set logits and attribute
logits. For board metrics, it averages corresponding learned triple scores and
per-card attribute logits, then chooses the highest-scoring triple at full raw
coverage. The ensemble path performs no confidence guard, abstention, symbol
solver, voting tie-break, or sealed-corpus access. It rejects missing or
incompatible output, mismatched triple ordering or logit lengths, and
non-finite logits.

`--ensemble-output-dir` is rejected during training and probe-only runs. An
invocation without the flag retains the existing single-model evaluator and
its default behavior.

This branch adds evaluator and parser tests only; it intentionally does not
run board-v7 research or sealed evaluation. Record any future measurement in
the board-v7 research log with both checkpoint hashes and the exact raw
evaluation allocation.
