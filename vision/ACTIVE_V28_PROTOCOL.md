# Active-v28 partial active-curriculum adaptation

Active-v27's aligned step-250 checkpoint improved the fresh paired browser result from 30 to 28
failures by removing four No-SET false clicks, but it added two Has-SET failures and exceeded the
one-board tolerance by one. Because v27 is a 250-step warm-started update with all shared,
count, and colour tensors frozen, coordinate interpolation with its exact active-v12 parent is a
meaningful test of adaptation strength. This protocol is locked before constructing any blend.

## Fixed candidates

Use active-v12 SHA-256 `ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec`
and active-v27 step-250 SHA-256
`77b46b40b9ae5fd4cd12606e6b78652199a5415bf9628b9e95169f13810bbf8f`. Construct three
strict single-checkpoint blends with v27 primary weights 0.25, 0.50, and 0.75. The blender must
reject key, shape, dtype, length, or finite-value mismatch. Verify that every tensor differing from
active-v12 remains inside the v27 shape/fill/texture/decision scope.

Evaluate all three on the unchanged exposed 4,096-board atlas. The active-v12 baseline remains 98
failures (41 Has-SET, 57 No-SET). A blend is eligible only with fewer than 98 failures, at most 45
Has-SET failures, and at most 61 No-SET failures. Select by fewest failures, then triple accuracy,
then lower v27 weight.

## Browser selection and audit

Compare only the best atlas-eligible blend against active-v12 on 512 paired fresh browser boards at
seed `2026090701`, using the frozen active-v13 head. The blend must have fewer total failures and
lose no more than one of 256 boards in either conditional. Use two ordered 256-board shards in fresh
browser processes, as established by the v27 infrastructure amendment. Fixed integration smoke is
mandatory after selection.

Only a selected blend receives one 1,000-board audit at seed `2026090799`, under the unchanged 98.0%
overall and 97.5% per-kind raw-action gates. Guarded accuracy remains diagnostic only. Any failure
restores active-v12 plus active-v13.

Because v27 proved that one Chrome process can exhaust its screenshot bridge after roughly 450
boards, execute this one logical audit as four ordered 250-board shards starting at 0/250/500/750,
each in a fresh browser process, and aggregate integer counts. This process-lifetime choice is
locked before opening the v28 audit seed and does not change its cases or gates.
