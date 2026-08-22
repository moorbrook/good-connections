# Active-v25 decision-head-only board fine-tuning

Active-v23/v24 show that board supervision can improve No-SET rejection, while updating the full
55,155-parameter stack reduces attribute/triple quality and eventually harms both board classes.
This protocol is locked before adding the freeze control.

## Controlled change

Retain the complete v24 training recipe: active-v10 warm start, differentiable twelve-card and
220-triple path, hardest-of-eight mining, alternating board classes, temperature 0.1, margin 0.25,
balanced dense triple loss, attribute/relation auxiliaries, data seed `590011`, learning rate
`8e-5`, and AdamW weight decay `1e-4`. Freeze the shared convolution, all four attribute
specialists, and the fill-texture branch. Optimize only parameters named `attribute_relation_*` or
`set_relation_*`. Strict checkpoint serialization still includes the complete unchanged model.

Run 1,000 steps and save snapshots at 250 and 1,000. Verify frozen parameters remain byte-identical
to the warm start and trainable parameters change before evaluation.

## Selection and audit

Use the unchanged atlas gate: fewer than 117 failures, at most 53 Has-SET failures, and at most 72
No-SET failures. The best eligible snapshot advances to browser seed `2026090401` against the
active-v12 deployment and frozen active-v13 head, with fewer total failures and no more than one
conditional loss per 256. Fixed integration smoke remains mandatory.

Only a selected candidate receives audit seed `2026090499` under the unchanged 98.0% overall and
97.5% per-kind raw-action gates. Guarded accuracy is diagnostic only.
