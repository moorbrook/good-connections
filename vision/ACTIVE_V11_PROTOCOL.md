# Active-v11 targeted curriculum protocol

Active-v10's fixed browser audit is already certain to miss its 98% gate: it exceeded the maximum
allowed failures before the stream finished. The run must still reach all 1,000 boards. Only after
completion is root seed `2026082099` retired and eligible to become development evidence.

## Development curriculum

Start from the 16 active-v10 development failures plus every completed active-v10 audit failure.
Then run 768 feedback-enabled boards with root seed `2026082101` and focus rate `0.75`. Focused
cases retain the failed case's theme, capture profile, and DPR while mutating viewport, visual seed,
and sometimes board seed; one quarter of focused cases are exact replay checks. Append every new
failure to the symbolic JSONL journal. Pixels remain transient: the production browser renderer
reconstructs each unique record and streams normalized card tensors directly to the trainer.

## Candidate sweep

Keep the active-v10 `p1-r050` checkpoint as an unchanged reference. Train four new checkpoints from
the same initialization and ordinary-data RNG stream used by active-v10:

| Candidate | Replay semantics | Batch rate | Per-label cap |
| --- | --- | ---: | ---: |
| `balanced-r050-p1` | one positive and one negative | 0.50 | 1 |
| `targeted-r050-p1` | only labels implicated by the observed failure | 0.50 | 1 |
| `targeted-r100-p1` | only labels implicated by the observed failure | 1.00 | 1 |
| `targeted-r100-p2` | only labels implicated by the observed failure | 1.00 | 2 |

Targeted replay means a No-SET false positive supplies its proposed hard negative, a Has-SET false
negative supplies a real positive, and a wrong proposed triple on a Has-SET board supplies both the
real positive and proposed hard negative. It does not invent an unaffected opposite-label replay.
All candidates otherwise use `DeployedVisualSetPolicy`, 10,000 MPSGraph steps, batch 64, AdamW,
augmentation on, negative log-sum-exp weight 2.5, data seed `590011`, and initialization seed
`3935559000370003845`.

First report the existing 4,096-board exposed atlas evaluation for every new checkpoint. Because
active-v10 showed that atlas rank does not reliably predict the browser, advance the best three new
candidates by atlas failures together with the unchanged active-v10 reference to a 512-board,
feedback-free browser development comparison at root seed `2026082102`. Select the fewest raw
failures, breaking ties by the higher worst conditional accuracy and then the atlas failure count.
This browser stream is selection data, not a holdout.

## Fresh browser audit

Build the extension with the selected single checkpoint and run exactly once on 1,000
feedback-free browser boards at root seed `2026082199`. The unchanged gate is at least 98.0% raw
overall and at least 97.5% raw for each board kind, all at 100% raw coverage. If it misses, retire
the stream and continue the active-learning loop under a newly predeclared seed.
