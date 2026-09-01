# Set — league play with solve telemetry

## What this is

A web app for playing the card game Set, aimed at a small group of
registered players organised into leagues. The distinguishing feature is
**high-resolution solve telemetry**: not just "who won", but how long each
player took to find each individual set, what mistakes they made along the
way, and how that compares to everyone else in the league who played the
**same board**.

Primary target is **iPhone Safari**, installed to the home screen as a PWA.
Desktop should work but is not the design target.

## Current scope

**Slice 1 (build this now): practice mode, single-player, no accounts.**

- Correct Set game logic, fully tested
- Both board modes (A and B)
- Faithful card rendering
- Local telemetry capture and an end-of-session summary
- Deployed as a static PWA, playable on iPhone

Practice mode *is* slice 1 — it is not additional scope on top. Leagues,
accounts, and scheduling are layers added over a working practice game.

Do not build auth, leagues, or a backend yet. Do not scaffold for them
"in advance" — but do keep game logic and telemetry capture in pure,
dependency-free modules so they can move to a server later without a
rewrite.

**Slice 2 (later): accounts + personal history.**
**Slice 3 (later): leagues, shared boards, cross-player comparison.**

If you think something in slice 1 is being designed in a way that will
block slices 2 or 3, say so before writing the code.

## Game rules — canonical

The deck is exactly 81 cards: every combination of four attributes, each
with three values.

| Attribute | Values |
|---|---|
| Count | 1, 2, 3 |
| Colour | red, green, purple |
| Shape | diamond, squiggle, oval |
| Fill | solid, striped, open |

Three cards form a **set** if, for *each* of the four attributes
independently, the three values are either all the same or all different.

Encode each attribute value as 0, 1, or 2. Then three cards form a set
if and only if, for every attribute, `(a + b + c) % 3 == 0`. Use this.
It is the whole detection algorithm and it should be three lines.

A useful consequence: any two cards determine a unique third card that
completes a set with them. This gives an O(n²) way to enumerate all sets
on a board, and a cheap way to test your brute-force O(n³) implementation
against a second independent implementation.

**Board dynamics (standard rules, for the free-play mode):** deal 12
cards. When a player takes a valid set, replace those three from the deck.
If no set exists among the visible cards, deal 3 more (15, then 18…).
Game ends when the deck is empty and no sets remain.

## Board generation and the puzzle format

**The game format is a fixed puzzle, not a deck playthrough.** Deal 12
cards. The board never changes — no replacement, no extra deals. The
player's job is to find every set on that board. This is what makes solve
times comparable between players, which is the entire point of the
telemetry.

**There are exactly two game modes.** A league is configured to use one
of them. Do not build a general "configure any number of sets" parameter.

**Mode A — six sets exactly.** The board contains exactly 6 sets. The
player is told this and can see how many remain.

**Mode B — unknown count.** An ordinary random 12-card deal. The player
is **not** told how many sets exist.

The modes differ in exactly one thing: whether the target count is
visible. Both end identically and automatically, the moment the last set
on the board is found. There is no "done" button, no way to concede, and
no way to end a game with sets remaining. In Mode B the player simply
does not know how close they are until the game ends underneath them.

The UI should show a running count of sets found in both modes. In Mode A
it is shown against a denominator (4 of 6); in Mode B it is a bare count
with no denominator.

### Generation

Both modes generate cheaply on demand — no precomputed board library.

- **Mode B:** shuffle, take 12. A board with zero sets must be rejected
  and redealt — with no "done" action, a setless board is a game that can
  never end. This is not an edge case to handle gracefully; it is a board
  that must never be dealt. About 3% of random deals.
- **Mode A:** rejection sampling. Shuffle, take 12, count sets, retry
  until the count is exactly 6. A random 12-card board averages 220/79 ≈
  2.78 sets, so 6 is above the mean but well within reach — expect on the
  order of tens of attempts, each only 220 triple-checks. Measure the
  actual attempt count and report it. If it is not fast, say so rather
  than silently changing approach.

For league play, generate the board **once** when the game slot is
created, store it, and serve that stored board to every member. Never
regenerate per player. Store the solution alongside it and **never send
the solution to the client.**

A curiosity about this generator, worth reading before changing the RNG:
because `mulberry32` advances its state by a constant, every seed walks the
same 2³² cycle and only picks a starting offset on it — so Mode A can reach
only ~5.3×10⁷ distinct boards, not the 2×10⁹ its seed space implies. Harmless
at league scale, and measured in [`docs/board-diversity.md`](docs/board-diversity.md).

## Practice mode

Alongside league play there is **practice**: unlimited puzzles, on
demand, in either Mode A or Mode B at the player's choice. No schedule,
no opponents, no cap on how many you play.

Practice games are fully instrumented. Same event log, same schema, same
storage, same derived statistics. A player should be able to see their
own practice history and watch their times move.

**Practice results never touch league scoring.** Two rules make that
hold:

1. Every game record carries an explicit
   `context: "league" | "practice"` field. There is **one** telemetry
   pipeline, not two — practice is a flag on the same records, never a
   parallel system that can drift out of sync with the real one.
2. Every league aggregate filters on it. This is the obvious place for a
   silent bug: someone writes a standings query, forgets the filter, and
   a player who practised forty puzzles quietly dominates the league.
   Make the context an explicit required argument to the aggregation
   functions rather than something with a default, so omitting it is a
   compile error and not a wrong number. Write a test that seeds practice
   games and asserts the standings do not move.

**Practice boards are generated fresh and never drawn from a league slot
that has not yet closed.** If a player could practise on the board that
is about to be their league game, the entire comparison is void. Practice
generation must be a separate call that has no access to the league board
table.

Practice is the natural place to reveal missed sets immediately on
abandonment — there is no one to leak to, and seeing what you missed is
the point of practising.

Keep the practice UI oriented around finishing a session and seeing a
summary, rather than an endless next-puzzle loop with a live-updating
stat line. Same information, and it ends somewhere.

### Note on comparing across modes
Mode B boards vary in difficulty — a 2-set board and a 5-set board are
different games. Within a single game slot every player gets the same
board so head-to-head comparison is valid, but season-long aggregates in
Mode B are noisier than Mode A. Do not mix modes in one leaderboard.

## Card rendering

The reference image is at `docs/reference-cards.png` — match it.

- Render cards as **SVG**, generated from the four attributes. No bitmap
  card images, no sprite sheets.
- One `<Card>` component takes `{count, colour, shape, fill}` and draws it.
  All 81 cards come from that one component.
- Shapes: diamond, squiggle, oval (stadium/rounded-rectangle). The
  squiggle is the fiddly one — it is a specific curved shape, not a
  generic blob. Build it as a bezier path and iterate against the
  reference until it reads right.
- Fills: solid, open (outline only), and striped (horizontal lines). Use
  an SVG `<pattern>` for the stripes, not manually drawn lines.
- Symbols are outlined in the card colour with a consistent stroke weight
  across all three fills.
- Layout: symbols centred, vertically stacked, evenly spaced, with the
  card's footprint identical regardless of count.

Touch targets must be comfortable on an iPhone. Selected cards need an
obvious, non-colour-dependent selected state (border/lift/scale), because
the card colours are already carrying meaning.

Optional, low priority: a colourblind-friendly palette toggle. The
standard red/green/purple is hard for some players. Do not change the
default.

## Telemetry

This is the reason the app exists. Treat it as a first-class feature, not
instrumentation bolted on afterwards.

Capture an append-only event log per game:

```
{ t_ms, type, payload }
```

where `t_ms` is milliseconds since board-reveal, and `type` is one of:

- `card_select` / `card_deselect` — which card
- `set_valid` — the three cards, plus which set index it was
- `set_invalid` — the three cards attempted
- `deal` — extra cards dealt (free-play mode)
- `abandon_prompt` — the give-up confirmation was opened
- `abandon_cancel` — the player backed out and kept playing
- `game_end` — with `reason: "completed" | "abandoned"`

### Abandoning

A game can be given up at any time. The control must be behind a
confirmation step — accidental abandonment would destroy a league entry
— and the confirmation should state what is being lost (elapsed time,
sets found so far).

**The clock keeps running while the confirmation is open.** If it paused,
the abandon prompt would be a free thinking break: open it, study the
board, cancel. Do not pause it.

Capture the near-misses. A player who opened the prompt at 4:32 and
backed out is a more interesting data point than one who never
considered it, and it costs nothing to record.

**Abandonment is not completion, and the scoring must not conflate them.**
An abandoned game has a *shorter* elapsed time than a completed one, so
any average-solve-time statistic that includes abandoned games actively
rewards giving up. Aggregate solve times over completed games only.
Abandonment feeds the completion-rate statistic, which is where it
belongs and is the reason that statistic is worth having.

**Do not reveal the missed sets on abandonment while the league slot is
still open.** Every member of a league plays the same board; showing the
solution to whoever quits first hands them something they can pass on.
Reveal after the slot closes, not before. In single-player practice
outside a league, reveal immediately — seeing what you missed is the
whole value.

Everything else is derived from this log, not stored separately: time to
first set, interval between consecutive sets, error count, error rate,
completion, total time. Derive on read. Do not compute and store summary
stats that could drift from the log.

Keep the log as plain JSON. In slice 1 it goes to `localStorage`; in
slice 3 the same shape is posted to the server. Design it now so that
transition is a change of transport only.

## Architecture

**Slice 1:**
- Vite + React + TypeScript, strict mode
- PWA manifest and service worker so it installs to the iPhone home screen
- Deployed static to Cloudflare Pages or Vercel
- No backend, no server to keep running

**Slice 2–3 (decided, not yet built):**
- Supabase for auth and Postgres, with row-level security
- **Boards are generated server-side.** League comparison is meaningless
  unless every player in the league receives a byte-identical board, and
  the client must never receive the deck order or the precomputed
  solution.
- Set validation stays client-side for instant feedback (it is trivially
  computable from the visible cards anyway, so this leaks nothing) with
  a server-side audit of the submitted event log.

Be honest about the anti-cheat ceiling: client timestamps are spoofable
by anyone willing to open devtools. The server can bound the window
between board-serve and submission, and that is enough for a league of
people who know each other. Do not build more than that.

## Working conventions

- **Game logic gets tests first.** Set detection, board generation, and
  set enumeration are pure functions with unambiguous correct answers.
  Write the tests, then implement. Vitest.
- Verify the deck is exactly 81 unique cards, that the O(n²) and O(n³)
  set enumerations agree, and that a known-set and known-non-set triple
  classify correctly.
- Game logic, board generation, and telemetry live in pure modules with
  no React imports.
- Small commits, one concern each. Commit before starting anything
  structural so I can roll back.
- Ask before adding a dependency. This project should stay close to
  vanilla; a card game does not need a state management library.
- If a requirement here is ambiguous, ask rather than guessing. The
  six-set board question above is the known one — there may be others.
