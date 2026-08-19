# FrankenTorch wasm32 compatibility patch

This directory vendors `crates/ft-runtime` from FrankenTorch commit
`dd6daf414e84462d2e0f35d8c103e1321e758b44`.

There is one semantic change: `now_unix_ms()` returns the deterministic value `0` on
`wasm32`. The upstream implementation calls `std::time::SystemTime::now()`, which traps on
`wasm32-unknown-unknown` while constructing `FrankenTorchSession`. Native builds retain the
upstream clock behavior. Browser inference does not persist or consume the internal evidence
timestamps, so operation ordering is preserved without inventing a JavaScript clock bridge.

The vendored source and license are retained so the patch is reviewable. Remove the Cargo
`[patch]` entries and this directory once the pinned upstream runtime provides a wasm-safe clock.
