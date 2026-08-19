#set page(
  width: 16in,
  height: 12.5in,
  margin: (x: 0.18in, y: 0.16in),
  fill: rgb("#FCFDFE"),
)
#set text(font: ("Helvetica Neue", "Arial"), fill: rgb("#152239"))
#set par(leading: 0.46em)

#let navy = rgb("#0E2F66")
#let ink = rgb("#152239")
#let muted = rgb("#61708A")
#let line = rgb("#C8D2E2")
#let blue = rgb("#2459C4")
#let teal = rgb("#0F8B78")
#let cyan = rgb("#159CB8")
#let purple = rgb("#7651C6")
#let orange = rgb("#D66B22")
#let rose = rgb("#D94B72")
#let green = rgb("#23945C")
#let red = rgb("#C63C3C")
#let gold = rgb("#C08B16")

#let badge(body, color: blue, fill: rgb("#EEF3FF")) = box(
  fill: fill,
  stroke: 0.5pt + color.lighten(62%),
  radius: 3pt,
  inset: (x: 5pt, y: 2.5pt),
  text(size: 5.7pt, weight: 700, fill: color, tracking: 0.045em, body),
)

#let number-badge(number, color) = circle(
  radius: 8.2pt,
  fill: color,
  align(center + horizon, text(size: 7.3pt, weight: 800, fill: white, str(number))),
)

#let section(number, title, body, color: blue, fill: white, height: auto) = block(
  width: 100%,
  height: height,
  fill: fill,
  stroke: 0.75pt + color.lighten(60%),
  radius: 6pt,
  inset: 7pt,
  [
    #grid(
      columns: (auto, 1fr),
      gutter: 5pt,
      align: (center + horizon, left + horizon),
      number-badge(number, color),
      text(size: 8pt, weight: 760, fill: color, tracking: 0.025em, title),
    )
    #v(5pt)
    #body
  ],
)

#let stage(title, detail, color: blue, fill: rgb("#F7F9FD")) = block(
  width: 100%,
  fill: fill,
  stroke: 0.55pt + color.lighten(65%),
  radius: 4pt,
  inset: 5pt,
  [
    #text(size: 6.2pt, weight: 720, fill: color)[#title]
    #v(1.5pt)
    #text(size: 5.35pt, fill: muted)[#detail]
  ],
)

#let tiny-row(label, value, color: ink) = grid(
  columns: (1fr, auto),
  gutter: 4pt,
  align: (left + horizon, right + horizon),
  text(size: 5.5pt, fill: muted, label),
  text(size: 5.5pt, weight: 670, fill: color, value),
)

#let flow(label: none, color: rgb("#7C8BA5")) = align(center + horizon)[
  #text(size: 12pt, fill: color)[$arrow.r$]
  #if label != none {
    v(-2pt)
    text(size: 4.5pt, fill: muted, label)
  }
]

#let card(mark, color, fill-kind: "outline") = box(
  width: 100%,
  height: 23pt,
  fill: rgb("#FFFFFF"),
  stroke: 0.55pt + rgb("#C8D2E2"),
  radius: 3pt,
  align(center + horizon, [
    #if fill-kind == "solid" {
      text(size: 13pt, weight: 800, fill: color, mark)
    } else if fill-kind == "stripe" {
      text(size: 12pt, weight: 650, fill: color, [#mark#h(1pt)#mark])
    } else {
      text(size: 13pt, weight: 450, fill: color, mark)
    }
  ]),
)

#let matrix(rows, cols, color: blue, active: 999) = grid(
  columns: (1fr,) * cols,
  rows: (7pt,) * rows,
  gutter: 1.3pt,
  ..range(rows * cols).map(i => box(
    width: 100%, height: 100%, radius: 1.2pt,
    fill: if i < active { color.lighten(48%) } else { rgb("#EEF1F6") },
    stroke: 0.25pt + color.lighten(58%),
  )),
)

#let branch(name, color, params) = block(
  width: 100%, height: 100%,
  fill: color.lighten(92%),
  stroke: 0.65pt + color.lighten(50%),
  radius: 5pt, inset: 5pt,
  [
    #align(center)[#badge(name, color: color, fill: color.lighten(90%))]
    #v(4pt)
    #stage([Conv2d], [12 -> 8 | k3 s2 p1 #linebreak() [N, 8, 6, 8]], color: color)
    #v(4pt)
    #stage([GELU $arrow.b$ PROJECTION], [flatten 384 -> 24 -> 3 #linebreak() GELU + softmax | #params params], color: color)
  ],
)

#let metric(value, label, color: blue) = block(
  width: 100%, height: 100%, fill: color.lighten(93%),
  stroke: 0.55pt + color.lighten(62%), radius: 4pt, inset: 5pt,
  align(center, [
    #text(size: 10pt, weight: 800, fill: color)[#value]
    #v(1pt)
    #text(size: 5pt, fill: muted)[#label]
  ]),
)

#let bar(label, ratio, value, color) = grid(
  columns: (44pt, 1fr, 35pt), gutter: 4pt,
  align: (left + horizon, left + horizon, right + horizon),
  text(size: 5.2pt, fill: ink, label),
  block(width: 100%, height: 7pt, fill: rgb("#EEF1F6"), radius: 2pt, inset: 0pt,
    box(width: ratio, height: 7pt, fill: color, radius: 2pt)),
  text(size: 5.2pt, weight: 700, fill: color, value),
)

// Header
#grid(
  columns: (1fr, 2.3fr, 1fr),
  align: (left + horizon, center + horizon, right + horizon),
  [#badge([LOCAL VISION-TO-ACTION POLICY], color: teal, fill: rgb("#EAF8F4"))],
  [
    #align(center)[#text(size: 18pt, weight: 820, fill: ink)[Set Vision Policy: Learned Perception to Browser Action]]
    #align(center)[#text(size: 7.5pt, fill: muted)[Compact CNN - four specialists - differentiable relation scorer - local WASM deployment]]
  ],
  [
    #align(right)[
      #badge([41,918 PARAMS], color: blue)
      #h(3pt)
      #badge([NO ATTENTION], color: orange, fill: rgb("#FFF3E9"))
    ]
  ],
)
#v(4pt)

// Top row: input, preprocessing, tensor representations, objectives
#grid(
  columns: (1.9fr, 4.2fr, 2.05fr, 2.05fr),
  rows: (2.50in,),
  gutter: 6pt,
  section(
    1, [INPUT OBSERVATION],
    [
      #text(size: 6pt, weight: 700, fill: ink)[Live board screenshot]
      #v(3pt)
      #grid(
        columns: (1fr,) * 4, gutter: 2.5pt, row-gutter: 2.5pt,
        card([$diamond.filled$], red, fill-kind: "solid"),
        card([$diamond$], green),
        card([$approx$], purple, fill-kind: "stripe"),
        card([$rect$], red),
        card([$diamond$], purple, fill-kind: "stripe"),
        card([$approx$], green, fill-kind: "solid"),
        card([$rect$], purple, fill-kind: "solid"),
        card([$diamond$], red),
        card([$approx$], red),
        card([$rect$], green, fill-kind: "stripe"),
        card([$diamond.filled$], green, fill-kind: "solid"),
        card([$approx$], purple, fill-kind: "solid"),
      )
      #v(5pt)
      #tiny-row([Screenshot], [H x W x 4 RGBA], color: cyan)
      #tiny-row([Cards], [12 clickable rectangles], color: cyan)
    ],
    color: teal,
    height: 100%,
  ),
  section(
    2, [DATA PREPROCESSING PIPELINE],
    [
      #grid(
        columns: (1fr, 13pt, 1fr, 13pt, 1fr, 13pt, 1fr, 13pt, 1fr),
        rows: (48pt,), gutter: 0pt,
        stage([Capture], [visible browser viewport], color: blue), flow(),
        stage([Crop], [12 DOM rectangles #linebreak() geometry only], color: blue), flow(),
        stage([Area average], [integer source bins #linebreak() anti-alias stable], color: blue), flow(),
        stage([Normalize], [RGB / 255 #linebreak() clamp 0..1], color: blue), flow(),
        stage([Pack], [12 x 1,890 #linebreak() f32 -> WASM], color: blue),
      )
      #v(6pt)
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr), gutter: 4pt,
        stage([SOURCE CROP], [variable W x H x 4], color: cyan),
        stage([CARD TENSOR], [[3, 21, 30]], color: cyan),
        stage([BOARD TENSOR], [[12, 3, 21, 30]], color: cyan),
        stage([CONTRACT], [browser = trainer fixture], color: cyan),
      )
      #v(6pt)
      #stage([PROCEDURAL TRAIN SOURCE], [Production Card renderer | four themes | multiple raster widths | seeded color, texture, geometry, and lighting variation], color: teal, fill: rgb("#EAF8F4"))
    ],
    color: blue,
    height: 100%,
  ),
  section(
    3, [INTERMEDIATE REPRESENTATIONS],
    [
      #grid(
        columns: (0.9fr, 1.25fr), gutter: 7pt,
        [
          #matrix(4, 6, color: cyan)
          #v(3pt)
          #align(center)[#text(size: 5pt, fill: muted)[RGB card]]
          #v(6pt)
          #matrix(5, 7, color: blue, active: 30)
          #v(3pt)
          #align(center)[#text(size: 5pt, fill: muted)[shared feature map]]
        ],
        [
          #tiny-row([Input], [[N, 3, 21, 30]], color: cyan)
          #tiny-row([Shared], [[N, 12, 11, 15]], color: blue)
          #tiny-row([Per head], [[N, 8, 6, 8]], color: rose)
          #tiny-row([Embedding], [[N, 24]], color: purple)
          #tiny-row([Attr logits], [[N, 4, 3]], color: orange)
          #v(5pt)
          #stage([BOARD CACHE], [12 cards encoded once #linebreak() 144 attribute logits], color: purple)
        ],
      )
    ],
    color: purple,
    height: 100%,
  ),
  section(
    6, [TRAINING OBJECTIVES],
    [
      #stage([SET LOSS], [CE([0, score], y-set)], color: orange, fill: rgb("#FFF8F1"))
      #v(4pt)
      #stage([AUXILIARY LOSS], [4 x mean CE(attribute logits, labels)], color: orange, fill: rgb("#FFF8F1"))
      #v(5pt)
      #align(center)[#text(size: 7pt, weight: 760, fill: orange)[$cal(L) = cal(L)_("set") + 4 cal(L)_("attr")$]]
      #v(5pt)
      #tiny-row([Optimizer], [AdamW | lr 8e-4 cosine | wd 1e-4], color: orange)
      #tiny-row([Budget], [20k x 64 = 1.28M triples], color: orange)
    ],
    color: orange,
    fill: rgb("#FFFCF8"),
    height: 100%,
  ),
)
#v(6pt)

// Middle row: representations, central architecture, heads + deployment
#grid(
  columns: (1.9fr, 6.25fr, 2.15fr),
  rows: (4.55in,),
  gutter: 6pt,
  section(
    8, [ROBUSTNESS & ANALYSIS],
    [
      #text(size: 5.7pt, weight: 720, fill: ink)[Procedural split protocol]
      #v(4pt)
      #stage([TRAIN], [272 atlases #linebreak() seeds 0-16 #linebreak() widths 128, 160, 224, 280], color: teal)
      #v(4pt)
      #stage([RESEARCH V5], [240 atlases #linebreak() seeds 11001-11002 #linebreak() 6 capture profiles + DPR 2], color: purple)
      #v(4pt)
      #stage([SEALED V5], [240 atlases #linebreak() seeds 12001-12002 #linebreak() unseen widths and profile mix], color: green)
      #v(7pt)
      #text(size: 5.35pt, fill: muted)[All splits span four themes. Research and final seeds and widths are absent from training. The sealed holdout is evaluated once, then retired.]
      #v(7pt)
      #stage([DIAGNOSTICS + GUARD], [Relation score/margin and attribute entropy/margin feed a research-fitted abstention gate. Repeated frames and post-click pixel change are also required.], color: rose, fill: rgb("#FFF6F8"))
      #v(6pt)
      #text(size: 5.15pt, weight: 680, fill: red)[Finite envelope claim - not proof of universal 100% reliability.]
    ],
    color: teal,
    height: 100%,
  ),
  section(
    4, [CORE ARCHITECTURE],
    [
      #grid(
        columns: (1.1fr, 15pt, 1fr, 15pt, 1.1fr),
        rows: (46pt,), gutter: 0pt,
        stage([CARD BATCH], [[N, 3, 21, 30] #linebreak() RGB f32], color: cyan),
        flow(),
        stage([SHARED CONV], [5 x 5 | stride 2 | pad 2 #linebreak() 3 -> 12 channels | 912 params], color: blue),
        flow(),
        stage([SHARED ACTIVATION], [exact-erf GELU #linebreak() [N, 12, 11, 15]], color: blue),
      )
      #v(5pt)
      #align(center)[#text(size: 5.5pt, weight: 700, fill: rose)[$arrow.b$ fan-out: same shared map enters four independent learned branches $arrow.b$]]
      #v(4pt)
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr),
        rows: (1.75in,), gutter: 5pt,
        branch([COUNT], cyan, [10,187]),
        branch([COLOR], green, [10,187]),
        branch([SHAPE], orange, [10,187]),
        branch([FILL], rose, [10,187]),
      )
      #v(6pt)
      #block(
        width: 100%,
        fill: rgb("#F7FBF9"),
        stroke: 0.7pt + green.lighten(58%),
        radius: 5pt, inset: 6pt,
        [
          #grid(
            columns: (1.05fr, 13pt, 1.15fr, 13pt, 1.05fr, 13pt, 0.95fr, 13pt, 0.85fr),
            rows: (52pt,), gutter: 0pt,
            stage([ENUMERATE], [12 choose 3 = 220 #linebreak() candidate actions], color: purple), flow(),
            stage([TRIPLE EVIDENCE], [for each attribute: #linebreak() p(i) + p(j) + p(k) #linebreak() shape [220, 4, 3]], color: green), flow(),
            stage([ATTRIBUTE MLP], [shared 3 -> 32 -> 1 #linebreak() GELU | x4 #linebreak() 161 params], color: green), flow(),
            stage([SET MLP], [concat 4 -> 16 -> 1 #linebreak() GELU #linebreak() 97 params], color: green), flow(),
            stage([ACTION], [argmax score #linebreak() -> [i, j, k]], color: teal),
          )
        ],
      )
    ],
    color: navy,
    height: 100%,
  ),
  grid(
    columns: (1fr,),
    rows: (2.00in, 1fr),
    row-gutter: 6pt,
    section(
      5, [ATTRIBUTE SPECIALIST OUTPUTS],
      [
        #grid(
          columns: (1fr, 1fr), gutter: 4pt, row-gutter: 4pt,
          stage([COUNT], [1 | 2 | 3 #linebreak() p = [p1, p2, p3]], color: cyan),
          stage([COLOR], [red | green | purple #linebreak() p = [pr, pg, pp]], color: green),
          stage([SHAPE], [diamond | squiggle | oval #linebreak() 3 learned logits], color: orange),
          stage([FILL], [solid | striped | outline #linebreak() 3 learned logits], color: rose),
        )
        #v(5pt)
        #text(size: 5.25pt, fill: muted)[Auxiliary labels make each branch inspectable. The relation scorer consumes probabilities, not argmax labels, so uncertainty and gradients remain continuous.]
        #v(4pt)
        #align(center)[#badge([SOFT SPECIALIZATION, LEARNED COMBINATION], color: purple)]
      ],
      color: purple,
      height: 100%,
    ),
    section(
      7, [INFERENCE & DEPLOYMENT],
      [
        #grid(
          columns: (1fr, 11pt, 1fr, 11pt, 1fr),
          rows: (43pt,), gutter: 0pt,
          stage([BROWSER], [capture + crop #linebreak() area resample], color: cyan), flow(),
          stage([WASM], [FrankenTorch #linebreak() embedded .ftsv], color: purple), flow(),
          stage([GUARDED ACTION], [two stable reads #linebreak() click + verify change], color: teal),
        )
        #v(5pt)
        #tiny-row([Checkpoint], [330 KB .ftsv], color: purple)
        #tiny-row([Generated WASM], [2.5 MiB], color: purple)
        #tiny-row([Cached inference], [2.502 ms / board], color: teal)
        #tiny-row([With pixel transfer], [2.767 ms / board], color: teal)
      ],
      color: purple,
      fill: rgb("#FCFBFF"),
      height: 100%,
    ),
  ),
)
#v(6pt)

// Bottom row: evaluation, benchmarks, model facts
#grid(
  columns: (3.75fr, 5.0fr, 1.55fr),
  rows: (2.40in,),
  gutter: 6pt,
  section(
    9, [BENCHMARKS & EVALUATION],
    [
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr), rows: (47pt,), gutter: 4pt,
        metric([487 / 4,096], [v5 research raw failures], color: purple),
        metric([1,085 / 10,000], [v5 sealed raw failures], color: orange),
        metric([99.84% @ 24.62%], [sealed guarded accuracy / coverage], color: green),
        metric([4 / 12], [guarded browser actions verified], color: teal),
      )
      #v(7pt)
      #grid(
        columns: (1.1fr, 1fr), gutter: 10pt,
        [
          #text(size: 5.6pt, weight: 720, fill: ink)[20k-step training time - about 20.9x faster]
          #v(4pt)
          #bar([CPU], 100%, [~1,500 s], muted)
          #v(3pt)
          #bar([MPSGraph], 4.79%, [71.9 s], blue)
        ],
        [
          #text(size: 5.6pt, weight: 720, fill: ink)[Board inference latency]
          #v(4pt)
          #bar([Cached], 90.4%, [2.502 ms], teal)
          #v(3pt)
          #bar([Transfer + infer], 100%, [2.767 ms], green)
        ],
      )
    ],
    color: blue,
    height: 100%,
  ),
  section(
    10, [AUTORESEARCH LOOP & PROMOTION GATES],
    [
      #grid(
        columns: (1fr, 12pt, 1fr, 12pt, 1fr, 12pt, 1fr, 12pt, 1fr),
        rows: (47pt,), gutter: 0pt,
        stage([HYPOTHESIS], [one falsifiable #linebreak() change], color: navy), flow(),
        stage([TRAIN], [fixed 20k-step #linebreak() budget], color: blue), flow(),
        stage([RESEARCH], [4,096 boards #linebreak() keep / discard], color: purple), flow(),
        stage([SEALED], [10,000 boards #linebreak() exactly once], color: green), flow(),
        stage([PROMOTE], [same checkpoint #linebreak() -> WASM], color: teal),
      )
      #v(7pt)
      #grid(
        columns: (1.1fr, 1fr), gutter: 8pt,
        [
          #text(size: 5.6pt, weight: 720, fill: ink)[Selection order]
          #v(3pt)
          #tiny-row([1], [fewer board failures], color: navy)
          #tiny-row([2], [higher triple accuracy], color: navy)
          #tiny-row([3], [lower latency, then size], color: navy)
        ],
        [
          #text(size: 5.6pt, weight: 720, fill: ink)[Reproducibility controls]
          #v(3pt)
          #tiny-row([Selected seeds], [490009 / 0x369dea0f31a53f85], color: purple)
          #tiny-row([Batch], [64 balanced triples], color: purple)
          #tiny-row([Replications], [8 fixed seed pairs], color: purple)
        ],
      )
    ],
    color: navy,
    height: 100%,
  ),
  section(
    11, [MODEL FACTS],
    [
      #tiny-row([Policy], [compact CNN], color: blue)
      #tiny-row([Attention], [none], color: orange)
      #tiny-row([Memory], [none], color: orange)
      #tiny-row([Runtime solver], [none], color: orange)
      #tiny-row([Output], [3 card indices], color: teal)
      #tiny-row([Parameters], [41,918], color: blue)
      #tiny-row([Actions], [220 / board], color: green)
    ],
    color: orange,
    fill: rgb("#FFFCF8"),
    height: 100%,
  ),
)
#v(6pt)

// Legend and source guide
#block(
  width: 100%, height: 1.08in,
  fill: rgb("#F7F9FC"), stroke: 0.65pt + line, radius: 5pt, inset: 7pt,
  [
    #grid(
      columns: (1.45fr, 1.7fr, 2.05fr, 2.15fr), gutter: 13pt,
      [
        #text(size: 5.6pt, weight: 760, fill: ink)[LEGEND - MODULE COLORS]
        #v(4pt)
        #grid(
          columns: (auto, 1fr, auto, 1fr), gutter: 4pt, row-gutter: 3pt,
          box(width: 9pt, height: 6pt, fill: blue.lighten(45%)), text(size: 4.9pt, fill: muted)[shared CNN],
          box(width: 9pt, height: 6pt, fill: rose.lighten(45%)), text(size: 4.9pt, fill: muted)[specialists],
          box(width: 9pt, height: 6pt, fill: green.lighten(45%)), text(size: 4.9pt, fill: muted)[relation scorer],
          box(width: 9pt, height: 6pt, fill: orange.lighten(45%)), text(size: 4.9pt, fill: muted)[loss / training],
          box(width: 9pt, height: 6pt, fill: purple.lighten(45%)), text(size: 4.9pt, fill: muted)[runtime / state],
          box(width: 9pt, height: 6pt, fill: teal.lighten(45%)), text(size: 4.9pt, fill: muted)[browser action],
        )
      ],
      [
        #text(size: 5.6pt, weight: 760, fill: ink)[TENSOR / SHAPE GUIDE]
        #v(4pt)
        #tiny-row([N], [card count or batch cards], color: navy)
        #tiny-row([C, H, W], [channels, height, width], color: navy)
        #tiny-row([[N, 4, 3]], [four attributes x three values], color: navy)
        #tiny-row([[220, 4, 3]], [all triple evidence vectors], color: navy)
      ],
      [
        #text(size: 5.6pt, weight: 760, fill: ink)[GRADIENT / DATA FLOW]
        #v(4pt)
        #text(size: 5pt, fill: muted)[$arrow.r$ solid: inference data flow #h(9pt) $arrow.b$ branch / supervision direction]
        #v(4pt)
        #text(size: 5pt, fill: muted)[Training loss backpropagates through both relation MLPs, probability sums, each specialist softmax and classifier, specialist convolutions, and the shared convolution. There is no nondifferentiable gate between loss and specialists.]
      ],
      [
        #text(size: 5.6pt, weight: 760, fill: ink)[IMPLEMENTATION SOURCES]
        #v(4pt)
        #text(size: 4.9pt, fill: muted)[Policy and browser path: `vision-policy/src/lib.rs` #linebreak() Fast training graph: `vision-mpsgraph/src/policy_graph.rs` #linebreak() Pixel contract: `extension/preprocess.js` #linebreak() Protocol and results: `vision/AUTORESEARCH.md`, `vision/RESEARCH_LOG.md`]
        #v(4pt)
        #text(size: 4.8pt, weight: 650, fill: ink)[Diagram generated from editable Typst source. Metrics measured on Apple M5 Pro, 2026-08-18.]
      ],
    )
  ],
)
