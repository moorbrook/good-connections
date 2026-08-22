# Active-v21 count-corrected multi-board negative groups

Active-v20 improved triple accuracy from 96.2524% to 97.6929%/98.5046% but catastrophically
weakened No-SET accuracy. Partitioning 32 negative logits into `g` independent log-sum-exp groups
reduces each aggregate by approximately `ln(g)` under comparable logit distributions. This
protocol is locked before adding or evaluating the correction.

## Controlled change

Retain the exact active-v20 sampler, topology, 10,000 steps, batch size 64, board-v8 corpus, data
seed `590011`, initialization seed `3935559000370003845`, attribute loss, and negative
log-sum-exp weight 2.5. For each group, add the non-learned constant `ln(g)` to its log-sum-exp
before negative BCE, then average group losses. Test four groups of eight and eight groups of four.
This is the unique count correction implied by splitting the old 32-term aggregate; it is not a
swept hyperparameter.

Require argument validation and one-step FrankenTorch/MPSGraph loss and shared-gradient parity for
four corrected groups.

## Selection and audit

Evaluate the fixed active-v10 control and both candidates on the exposed 4,096-board atlas stream.
Advance the best corrected candidate only if it has fewer than 117 total failures and neither
conditional exceeds the control by more than 4/2,048 boards (at most 53 Has-SET and 72 No-SET
failures). Then compare it with the current active-v12 deployment using the frozen active-v13 head
on 512 feedback-free browser boards at seed `2026083101`. It must have fewer total failures and lose
no more than one of 256 boards in either conditional. Fixed integration smoke is mandatory before
any audit.

Only a selected candidate receives one 1,000-board feedback-free audit at seed `2026083199`, with
the unchanged 98.0% overall and 97.5% per-kind raw-action gates. Guarded accuracy remains
diagnostic only.
