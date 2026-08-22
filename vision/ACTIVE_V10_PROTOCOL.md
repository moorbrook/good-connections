# Active-v10 robustness protocol

This development cycle tests whether browser-mined failure replay improves the deployed visual
policy without hiding regressions behind abstention or an oracle. It changes the training
distribution only; topology, optimizer, base corpus, initialization, and evaluation policy remain
fixed.

## Development curriculum

- Miner root seed: `2026082001`.
- Boards: 256, exactly balanced Has-SET/No-SET.
- Focus rate: zero; every case is fresh rather than descended from an earlier failure.
- Persistent artifact: `vision-artifacts/active-v10/development-curriculum.jsonl`.
- Observed before training: 16 failures, with 8 Has-SET and 8 No-SET failures.

The 256-board miner result is development data. It may diagnose and train candidates but cannot be
reported as a holdout result.

## Controlled rate sweep

Train five models from scratch with curriculum rates `0`, `0.02`, `0.05`, `0.10`, and `0.20`.
All use:

- production `DeployedVisualSetPolicy` (55,155 parameters);
- `vision-artifacts/board-v8-training-atlases` as the unchanged base corpus;
- 10,000 MPSGraph optimizer steps, batch size 64, AdamW, and existing augmentation;
- negative log-sum-exp weight 2.5;
- data seed `590011` and initialization seed `3935559000370003845`.

The ordinary-data RNG and curriculum RNG are independent. Every candidate generates the same base
batch at every optimizer step; a selected curriculum step replaces that already-generated batch.
Thus rate is the only controlled sampling difference.

Evaluate each checkpoint at 100% coverage on the same exposed development distribution:

- `vision-artifacts/board-v8-research-atlases`;
- 4,096 exactly mixed boards;
- board seed `324222009835245`;
- 16,384 triples with triple seed `327145488695021`.

Select the fewest raw board failures, breaking a tie by triple accuracy. Do not use guard coverage,
confidence thresholds, or the 256 mined boards for selection.

## Fresh browser audit

After selecting one rate, rebuild the extension with that single checkpoint and run exactly once on
1,000 feedback-free browser boards with root seed `2026082099`. The gate is:

- at least 98.0% raw accuracy overall;
- at least 97.5% raw accuracy for both Has-SET and No-SET;
- 100% raw coverage (abstention does not change the raw score).

If the winner misses the gate, the audit is retired. Its failures may enter the next development
curriculum, but the next iteration must use a new predeclared audit seed.

## Phase-one result and sparse replay amendment

The full-batch rate sweep completed before this amendment. Rate zero reproduced the deployed
checkpoint byte-for-byte and scored 117 failures. Rates `0.02/0.05/0.10/0.20` scored
`127/153/151/157` failures respectively. Full-batch replacement is rejected: even the smallest rate
over-concentrates one failed board's gradient and reduces generalization.

Before further training, a sparse within-batch sweep is added. It preserves every generated base
batch and replaces only a small, label-balanced number of its samples when curriculum is selected:

| Candidate | Batch selection rate | Replacements per label | Maximum replaced samples |
| --- | ---: | ---: | ---: |
| `p1-r025` | 0.25 | 1 | 2/64 |
| `p1-r050` | 0.50 | 1 | 2/64 |
| `p1-r100` | 1.00 | 1 | 2/64 |
| `p2-r100` | 1.00 | 2 | 4/64 |

All other training and development-evaluation controls remain unchanged. Selection compares these
four candidates with the byte-identical rate-zero control using the original fewest-failures rule.
The browser audit remains unopened.
