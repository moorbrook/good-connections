# Active-v12 aligned checkpoint-soup protocol

Active-v11's fresh audit exceeded 20 failures before completion, making its 98% overall gate
mathematically unreachable. The audit must still finish. This protocol is fixed before creating or
evaluating any checkpoint soup.

## Fixed ingredients

Blend the aligned parameters of exactly two single-model checkpoints:

- active-v10 reference: `b2e763d5bf3604734dc3fe491ba47d0ef47dc2bc53e23d1fdf35cb25ca146b2e`;
- active-v11 targeted `r100-p2`:
  `52036c615e1c6a695996d7d184611ea4f41e9e68cd06bdc659a50935465494d4`.

They have identical topology, initialization, ordinary-data RNG stream, optimizer schedule, and
base batches. A soup is a key-by-key linear interpolation of parameters, not an output ensemble and
not additional training. The tool must reject mismatched keys, shapes, dtypes, or non-finite values.

Create three candidates with active-v11 primary weights `0.25`, `0.50`, and `0.75`; the endpoints
remain the two original checkpoints. Evaluate all three on the exposed 4,096-board atlas stream used
by active-v10/v11. Advance the best two soups by atlas failures, together with both endpoints, to a
512-board feedback-free browser development comparison at root seed `2026082201`. Select the fewest
raw failures, breaking ties by worst conditional accuracy and then atlas failures.

## Fresh browser audit

Embed the selected single checkpoint and run exactly once on 1,000 feedback-free browser boards at
root seed `2026082299`. The unchanged gate is at least 98.0% raw overall and at least 97.5% raw for
both Has-SET and No-SET at 100% raw coverage. If it misses, retire the audit and move beyond replay
and aligned parameter averaging rather than extending this sweep post hoc.
