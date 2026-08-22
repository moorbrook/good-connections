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

The project uses compact learned convolutional policies, not a transformer.
`DeployedVisualSetPolicy` is the canonical native and browser topology.
`ResearchVisualSetPolicy`, `TypedRelationResearchPolicy`, and
`WideTypedRelationResearchPolicy` are explicitly named experiment controls. None has attention, a
language model, a token sequence, or a recursive-language-model loop. They are therefore neither
VLMs nor general VLAs. In the narrow control sense each is a vision-to-action policy: its action is
three card indices or the explicit `NoSet` action.

The shipped 55,155-parameter `DeployedVisualSetPolicy` topology is:

1. Area-resample each card screenshot to 30×21 RGB.
2. Apply a 5×5 stride-2 convolution from 3 to 12 channels and GELU.
3. Fork into four attribute-specialist branches. Each applies its own 3×3 stride-2 convolution from
   12 to 8 channels, a rate-two dilated 3×3 residual convolution, and GELU.
4. In each branch, flatten the 8×6×8 feature map, project it to 24 features with GELU, and predict
   one group of three attribute logits.
5. Run a separate raw-RGB fill-texture path—3×3 stride 1, two 3×3 stride-2 convolutions, and a
   384→24→3 head—and add its output only to the fill logits. It cannot directly alter count,
   colour, or shape.
6. Softmax each attribute, sum the three cards' distributions, and classify each attribute as
   `same`, `different`, or `invalid` with a shared 3→32→3 MLP.
7. Combine the twelve continuous relation probabilities with a 12→16→1 MLP.

GELU is only the smooth nonlinearity between learned affine operators; it does not provide attention
or memory. Each board card is encoded once, then the learned permutation-invariant relation scorer
evaluates all 220 triples on a 12-card board. A non-negative highest logit selects that triple; a
negative highest logit produces `NoSet`, which the extension displays without clicking.

This repository uses constrained **model search**, not a scaling-law study and not runtime program
synthesis. Experiments change one falsifiable aspect of the policy topology, operator choice,
inductive bias, training distribution, or input contract. Width and depth may be candidates, but a
few small runs cannot establish a scaling law. Candidates are ranked lexicographically by board
failures, triple accuracy, measured latency, WASM size, and simplicity.

## Implementation boundaries

| Concern | Owner | Consumers |
| --- | --- | --- |
| Topology, dimensions, relation scoring, board action enumeration | `vision-policy/` | trainer and WASM runtime |
| Parameters | topology-specific FrankenTorch `.ftsv` checkpoints | research evaluator or embedded browser runtime |
| Training and offline evaluation | `vision/` | autoresearch loop |
| MPS training graph | `vision-mpsgraph/` | accelerated training only |
| Browser ABI only | `vision-wasm/` | extension |
| DOM capture, card rectangles, clicks | `extension/` | live browser |

There is no handwritten browser copy of convolution, GELU, softmax, or relation scoring. WebAssembly
constructs `DeployedVisualSetPolicy` and strictly loads its embedded checkpoint. Native training and
evaluation now construct that same topology by default; `--legacy-untyped` and `--wide-vision` are
explicit research alternatives. The MPSGraph trainer duplicates the selected topology in a
training-only graph and exports the same `.ftsv` format; fixed-batch forward/loss/gradient parity
checks guard that duplication. Strict loading rejects checkpoints from incompatible policy types.

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

Historical Board-v5 validation used unseen renderer seeds, widths, DPR 1/2 captures, and six capture
profiles: clean, blur, crop/zoom-in, shadow/scale-down, low contrast, and JPEG recompression. Its
separate sealed corpus was rendered and hashed before training and evaluated exactly once after the
recipe and guard were frozen. The current raw model is not universally reliable under scale shifts;
the extension therefore abstains conservatively instead of presenting a raw argmax as safe.

Historical Board-v6 tested two raw-model architecture changes on fresh research seeds at 100% action
coverage. The 41,918-parameter control scored 86.82% (540 failures in 4,096 boards). A spatial-pyramid specialist
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
`ft-serialize` remains the only checkpoint format. Native training, native evaluation, and WASM
inference use `DeployedVisualSetPolicy` by default. `wasm-bindgen` supplies JavaScript glue; it does
not reimplement model operators.

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

Promote a validated deployed-topology checkpoint to `vision-artifacts/model/set-vision.ftsv`, then
build and benchmark:

```sh
tools/build-vision-extension.sh
node tools/benchmark-vision-wasm.mjs 100
node tools/calibrate-vision-guard.mjs \
  vision-artifacts/autoresearch-v9/dev-fill-texture-wide-r1-10k/abstention-samples.tsv 0.999
```

The embedded checkpoint is the selected single fill-texture replica, not the four-checkpoint NumPy
contextual/hypergraph ensemble. On 8,192 disjoint 190xx boards it made 7,987 correct raw decisions:
**97.4976% at 100% coverage**. The split was 4,012/4,096 (97.9492%) on boards with a SET and
3,975/4,096 (97.0459%) on `NoSet` boards.

The browser guard is reported separately as counts rather than used to inflate the model headline.
On the same stream it allowed 2,371 clicks: 2,363 correct and 8 wrong. On boards with a SET it
clicked 2,369/4,096 (57.8369% click coverage), with 6 wrong clicks (99.7467% click accuracy). It also
allowed 2 wrong clicks on `NoSet` boards, so it is not a fail-safe. In the rebuilt extension, all 12
responsive browser proposals were valid; 7 passed the guard, completed their clicks, and verified a
pixel-level board transition. A 1,000-iteration Node/WASM benchmark measured 5.62 ms
cached inference and 5.73 ms including pixel transfer per 12-card board on the current host.

## Generative streaming evaluation and active curriculum

The browser harness can generate an unbounded balanced stream of fresh `HasSet` and `NoSet`
boards. Every iteration randomizes the board seed, per-card visual seed, theme, capture profile,
responsive viewport, and device scale. The extension receives only live screenshot pixels and element
geometry. The harness reads card IDs only after the frozen action, when it independently scores the
prediction.

Mine failures continuously:

```sh
npm run test:vision-stream -- --seed 20260819
```

The miner writes each failure as a reproducible JSONL record to
`vision-artifacts/stream-curriculum.jsonl`. Later iterations sample both fresh cases and cases from
that growing curriculum; one quarter of focused cases are exact replays and the rest mutate the
failed conditions. `--focus-rate 0.5` controls the fresh/focused mixture. The record stores the board
and visual seeds, theme, viewport, DPR, board kind, and post-prediction oracle result, so a training
cycle can lazily regenerate pixels in memory without retaining screenshots.

Feed the mined failures into a new MPSGraph training run without creating an image corpus:

```sh
npm run build
cargo build --release --manifest-path vision-mpsgraph/Cargo.toml
set -o pipefail
node tools/render-vision-boards.mjs \
  --curriculum vision-artifacts/stream-curriculum.jsonl \
| vision-mpsgraph/target/release/set-vision-mpsgraph \
  --steps 10000 --negative-logsumexp-loss-weight 2.5 \
  --curriculum-stdin --curriculum-rate 0.5 \
  --curriculum-examples-per-label 1 --targeted-curriculum \
  --output vision-artifacts/model-active/set-vision.ftsv
```

The browser regenerates each failed board with the production React/SVG renderer, applies the same
screenshot cropping and area resampling as the extension, and streams normalized card tensors over
stdout. The Rust trainer retains those tensors only in memory. `--curriculum-rate` is the probability
that an optimizer step receives replay examples; it is not an evaluation weight.
`--curriculum-examples-per-label` caps how many slots replay can replace in each label half. The
default mixer replaces both labels symmetrically. `--targeted-curriculum` instead follows the
observed error: a No-SET false positive supplies its proposed hard negative, a Has-SET false
negative supplies a real positive, and a wrong proposal on a Has-SET board supplies both. The same
flags and semantics are supported by the FrankenTorch reference trainer.

Run a finite scoring-only audit with no curriculum input or feedback:

```sh
npm run test:vision-stream -- \
  --audit --boards 10000 --seed 20260820 --report-every 250 \
  --min-raw-accuracy 0.97
```

Miner accuracy is adaptive development telemetry and must not be reported as a holdout metric. Only
an audit stream whose seed was fixed before evaluation is eligible for a generalization claim. A
failure from an audit may be copied into the next development curriculum only after that audit has
been retired.

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
still score confidently, as the disjoint accepted errors demonstrate.

## Verification

```sh
npm run build
npm test
node tools/test-vision-preprocessing.mjs
node tools/test-vision-guard.mjs
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
