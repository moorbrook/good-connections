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

Board-v5 research validation uses unseen renderer seeds, widths, DPR 1/2 captures, and six capture
profiles: clean, blur, crop/zoom-in, shadow/scale-down, low contrast, and JPEG recompression. Its
separate sealed corpus was rendered and hashed before training and evaluated exactly once after the
recipe and guard were frozen. The current raw model is not universally reliable under scale shifts;
the extension therefore abstains conservatively instead of presenting a raw argmax as safe.

Board-v6 tested two raw-model architecture changes on fresh research seeds at 100% action coverage.
The 41,918-parameter control scored 86.82% (540 failures in 4,096 boards). A spatial-pyramid specialist
regressed to 86.52% (552 failures) and was rejected. A parallel 3×3+5×5 specialist improved to
87.38% (517 failures), mainly by reducing shadow errors, but worsened crop errors and remained far
from the zero-failure promotion gate. The board-v6 sealed corpus therefore remains evaluator-
unopened, neither candidate was deployed, and no abstention model participated in selection. Exact
metrics are in `vision/BOARD_V6_RESULTS.tsv`.

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

Eight independent board-v5 training replications exposed substantial seed sensitivity. The selected
unchanged-topology replicate had 487 failures on 4,096 research boards (88.11% raw board accuracy).
Its one permitted 10,000-board sealed evaluation had 1,085 failures (89.15%). A research-only linear
guard over relation score, score margin, attribute entropy, and weakest attribute margin accepted
24.62% of sealed boards with 99.84% accepted accuracy (4 accepted errors). It was not retuned after
the sealed result. In the rebuilt extension, all 12 responsive browser proposals were valid; 4 passed
the guard and completed click plus pixel-change verification, while 8 safely abstained. A
1,000-iteration Node/WASM benchmark measured 2.502 ms cached inference and 2.767 ms including pixel
transfer per 12-card board.

## Load and use the extension

1. Open `chrome://extensions`, enable Developer mode, and choose **Load unpacked**.
2. Select this repository's `extension/` directory.
3. Open a game board and click the extension icon to show the side panel.
4. Keep `.card-btn` for this app, or enter the live board's selector for clickable cards.
5. Use **Scan board** first. **Play best** remains disabled when diagnostics or repeated-frame
   agreement fail.

Guarded auto-play requires the frozen diagnostic model, two stable captures with the same action,
and a stable material board change after clicking. The extension requests
`activeTab`, not blanket host access; the broader host permission in the headless integration test
exists only in a temporary copied manifest because CDP clicks cannot grant `activeTab`.
The displayed values are raw diagnostics, not a correctness probability: a perception error can
still score confidently, as the four accepted sealed errors demonstrate.

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
