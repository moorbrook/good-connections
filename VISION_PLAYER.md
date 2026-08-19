# Learned visual Set player

This project contains a small local computer player for a Set-style card game. A board contains
cards with four independent three-valued attributes: count, colour, shape, and fill. A valid move
selects three cards for which every attribute is either all the same or all different.

For a human, the game primarily tests visual discrimination, compositional pattern recognition,
and search across candidate triples. Perception is essential; working memory helps avoid rescanning,
but long-term memory is not required because each board can be solved from its current pixels.

The extension follows the same observable interface: screenshots in, three card positions out. It
does not read card labels, React state, solutions, or network payloads.

## Honest model description

`VisualSetPolicy` is a compact learned convolutional policy, not a transformer. It has no attention,
language model, token sequence, or recursive-language-model loop. It is therefore neither a VLM nor
a general VLA. In the narrow control sense it is a vision-to-action policy: its action is the three
card indices to select.

The 41,918-parameter topology is:

1. Area-resample each card screenshot to 30×21 RGB.
2. Apply a 5×5 stride-2 convolution from 3 to 12 channels and GELU.
3. Fork into four attribute-specialist branches. Each applies its own 3×3 stride-2 convolution from
   12 to 8 channels and GELU.
4. In each branch, flatten the 8×6×8 feature map, project it to 24 features with GELU, and predict
   one group of three attribute logits.
5. Softmax each attribute, sum the three cards' distributions, and score each attribute with a
   shared 3→32→1 MLP.
6. Combine the four relation features with a 4→16→1 MLP.

GELU is only the smooth nonlinearity between learned affine operators; it does not provide attention
or memory. Each board card is encoded once, then the learned permutation-invariant relation scorer
evaluates all 220 triples on a 12-card board. The highest logit becomes the three-position action.

This repository uses constrained **model search**, not a scaling-law study and not runtime program
synthesis. Experiments change one falsifiable aspect of the policy topology, operator choice,
inductive bias, training distribution, or input contract. Width and depth may be candidates, but a
few small runs cannot establish a scaling law. Candidates are ranked lexicographically by board
failures, triple accuracy, measured latency, WASM size, and simplicity.

## Implementation boundaries

| Concern | Owner | Consumers |
| --- | --- | --- |
| Topology, dimensions, relation scoring, board action enumeration | `vision-policy/` | trainer and WASM runtime |
| Parameters | one FrankenTorch `.ftsv` checkpoint | native evaluation and embedded browser runtime |
| Training and offline evaluation | `vision/` | autoresearch loop |
| MPS training graph | `vision-mpsgraph/` | accelerated training only |
| Browser ABI only | `vision-wasm/` | extension |
| DOM capture, card rectangles, clicks | `extension/` | live browser |

There is no handwritten browser copy of convolution, GELU, softmax, or relation scoring. Both native
evaluation and WebAssembly call `VisualSetPolicy::score_board` through FrankenTorch. The MPSGraph
trainer temporarily duplicates the topology in a training-only graph and exports the same `.ftsv`
format; fixed-batch forward/loss/gradient parity checks guard that duplication.

Two boundaries intentionally have separate adapters:

- Rust reads PNG atlases while JavaScript reads browser RGBA screenshots. Both implement the same
  integer-bin area-average contract, checked by matching fixtures.
- The application owns its TypeScript game rules; the trainer owns an offline Rust label generator;
  and the browser test owns a small independent oracle. These are separate trust domains. The
  deployed extension contains no symbolic Set solver.

The dense atlas and responsive test board are also intentionally different layouts. Both reuse the
production `Card` renderer: the atlas makes exhaustive procedural capture efficient, while the
responsive board exercises the deployment geometry.

## Procedural data and evaluation

`src/vision/Atlas.tsx` renders all 81 cards with seeded changes to brightness, contrast, hue,
saturation, spacing, stripe period, stroke, rotation, scale, and vertical position. The capture tool
also renders several source raster widths before every card is reduced to 30×21. Training samples
theme and source-width buckets uniformly, preventing a large fixed-width corpus from silently
dominating the responsive examples.

Research validation uses themes, renderer seeds, and source widths outside the training corpus. A
candidate must reach zero wrong selections on the fixed 4,096-board evaluator, then pass a separately
generated final holdout exactly once. A finite zero-failure result describes the tested envelope; it
is not proof of universal 100% reliability.

Generate atlases and run one model-search experiment with:

```sh
tools/render-vision-atlases.sh vision-artifacts/atlases 17 0 all 128,160,224,280
tools/render-vision-atlases.sh vision-artifacts/validation-v2-atlases 8 9001 all 104,120,152,192,216,272,344,376
tools/run-vision-mps-experiment.sh <tag> "<one falsifiable hypothesis>"
```

Do not reuse validation or final-holdout seeds as training data. See `vision/AUTORESEARCH.md` for the
search contract and promotion rules.

## Training backends, FrankenTorch, and WebAssembly

Reference CPU training, native evaluation, and browser inference use FrankenTorch directly, pinned
to commit `dd6daf414e84462d2e0f35d8c103e1321e758b44`. Architecture-search training can instead use the
direct MPSGraph Rust crate in `vision-mpsgraph/`. MPSGraph is a training accelerator only; it is not
a second browser inference runtime. The accelerated trainer exports through FrankenTorch's native
state-dict serializer, and the native evaluator must accept and verify every exported checkpoint.
`ft-serialize` remains the only checkpoint format, and the WASM crate loads that same state into the
shared policy. `wasm-bindgen` supplies JavaScript glue; it does not reimplement model operators.

The controlled specialist recipe runs with:

```sh
cd vision-mpsgraph
cargo run --release -- \
  --steps 20000 \
  --no-augment \
  --output ../vision-artifacts/<candidate>/set-vision.ftsv
```

On an Apple M5 Pro, the promoted 20,000-step run took 60.908 seconds at 328.36 steps/s, including
procedural sampling, tensor conversion and transfer, forward/backward, gradient readback, and CPU
AdamW. The comparable FrankenTorch CPU run took approximately 25 minutes, so the training phase is
about 24.6× faster. MPSGraph's Objective-C
temporaries must remain inside the per-step autorelease pool; omitting it caused an invalid run to
exhaust Metal memory after 16,300 steps.

The repository has one pinned Rust toolchain in the root. `vendor/frankentorch-runtime/` is an exact
copy of the pinned runtime with one documented compatibility change: on `wasm32`, its internal
evidence timestamp is the deterministic value zero instead of calling `SystemTime::now()`, which
traps on `wasm32-unknown-unknown`. Native clock behavior is unchanged. Remove the patch when upstream
provides a WASM-safe clock.

Promote a validated checkpoint to `vision-artifacts/model/set-vision.ftsv`, then build and benchmark:

```sh
tools/build-vision-extension.sh
node tools/benchmark-vision-wasm.mjs 100
```

The promoted checkpoint produced 0 failures on 4,096 board-v4 research boards and then 0 failures
in its one permitted 10,000-board sealed evaluation. In the rebuilt extension, all 12 responsive
screenshot-to-click cases passed. A 1,000-iteration Node/WASM benchmark measured 2.416 ms cached
inference and 2.508 ms including pixel transfer per 12-card board. These are finite measurements of
the tested renderer and machine, not a universal correctness or latency guarantee.

## Load and use the extension

1. Open `chrome://extensions`, enable Developer mode, and choose **Load unpacked**.
2. Select this repository's `extension/` directory.
3. Open a game board and click the extension icon to show the side panel.
4. Keep `.card-btn` for this app, or enter the live board's selector for clickable cards.
5. Use **Scan board** first. **Play best** remains disabled below the configured confidence.

Guarded auto-play stops when confidence falls below the threshold. The extension requests
`activeTab`, not blanket host access; the broader host permission in the headless integration test
exists only in a temporary copied manifest because CDP clicks cannot grant `activeTab`.
Confidence is a model logit, not a correctness guarantee: a perception error can still be confident,
which is why promotion depends on board-level holdouts rather than the threshold alone.

## Verification

```sh
npm run build
npm test
node tools/test-vision-preprocessing.mjs
node tools/test-vision-extension.mjs
cargo clippy --manifest-path vision-policy/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path vision-policy/Cargo.toml
cargo clippy --manifest-path vision/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path vision/Cargo.toml
cargo clippy --manifest-path vision-wasm/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path vision-wasm/Cargo.toml
```

The headless extension test uses a real CDP-rendered PNG at the one screenshot-acquisition boundary
because Chrome 151 leaves `captureVisibleTab()` pending in headless mode. The extension's rectangle
collection, crop, resampling, FrankenTorch inference, highlighting, and action selection remain the
production paths.
