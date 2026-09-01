# Why there are only ~53 million Mode A boards

*A curiosity, not a bug report. Nothing here needs fixing today — but it is a
nice example of a generator being much smaller than it looks, and it took a
falsifiable prediction to pin down.*

## The question

> I'm in four leagues, all Mode A, possibly with different reset times. Can I be
> sure no two of them deal the same board on the same day?

No. Nothing guarantees it — and the honest probability is ~38× worse than the
seed space suggests.

## How seeds are assigned

`today_puzzle()` in [`supabase/schema.sql`](../supabase/schema.sql) draws one
independent random seed per `(league, local date)`:

```sql
insert into daily_seeds (league_id, puzzle_date, seed)
  values (p_league, d, (random() * 2000000000)::bigint)
  on conflict (league_id, puzzle_date) do nothing;
```

There is no cross-league uniqueness check anywhere — no constraint, no retry,
nothing that has ever looked at another league's seed. Collision avoidance is
purely probabilistic.

**Reset times are irrelevant to this.** A league's timezone only decides what
`d` is; the seed is drawn at random regardless. Two leagues with identical reset
times are no more or less likely to collide than two with different ones.

The naive reading is therefore "2×10⁹ seeds, so ~1 in 2 billion per pair". That
is wrong, and measurably so.

## The mechanism

`mulberry32` advances its state by a **constant**:

```js
a = (a + 0x6d2b79f5) | 0
```

That constant is odd, so multiplication by it is invertible mod 2³². Write
`M = 0x6d2b79f5`. The state visited at step *k* from seed *s* is `s + k·M`, which
is `M·(u + k)` where `u = s·M⁻¹ mod 2³²`.

So **every seed walks the same single cycle of length 2³²**. The seed does not
choose a different sequence; it chooses a *starting offset* on one global one.

Now layer the generator on top. A Mode A deal is a Fisher–Yates shuffle of 81
cards, which consumes exactly **80 draws**, and rejection sampling re-deals until
a board has exactly six sets. So a seed's board is:

> the first *accepting* 80-draw block at or after my offset

Two seeds whose offsets are congruent mod 80 with no accepting block between them
land on the **same block** and produce a byte-identical board. Since only 2.349%
of random 12-card deals have exactly six sets (1 in 42.6 — the suite's own
`board.test.ts` independently measures 44.1 deals per board), each accepted board
is fed by a run of ~43 consecutive seeds on that lattice.

## The falsifiable prediction

If the above is right, then seed `s` and seed `s + 80·M mod 2³²` shift the offset
by exactly one block, and so must give the **identical board** whenever `s`
rejected its first block. And a shift of `1·M` — one draw, not a whole block —
must *not* align.

Both hold:

```
seed 1000 (accepted on attempt 82) vs seed 496377976 (attempt 81) -> IDENTICAL BOARD
seed 1001 (accepted on attempt 30) vs seed 496377977 (attempt 29) -> IDENTICAL BOARD
seed 1002 (accepted on attempt 17) vs seed 496377978 (attempt 16) -> IDENTICAL BOARD
394/394 identical => CONFIRMED
control, shift by 1 draw instead of 80: 0/100 identical
```

Note the attempt counts stepping down by exactly one — that is the offset walking
forward one block at a time toward a fixed accepting block. Seed 1000's board is
produced by 82 of the 200 seeds `1000 + m·80·M`, matching its attempt count of 82.

## The measurement

Over 400,000 seeds drawn exactly as Postgres draws them:

| | |
|---|---|
| Duplicate seeds | 58 |
| Distinct-seed board collisions | **1,501** |
| Expected from the 2×10⁹ seed space alone | 40 |
| **Effective distinct Mode A boards** | **5.33×10⁷** |
| Seed space overstates diversity by | **38×** |

For scale, C(81,12) = 7.07×10¹³ twelve-card subsets, of which ~2.349% have exactly
six sets, so there are ~1.66×10¹² Mode A boards in principle. **About 0.003% of
them are reachable at all.** The other 99.997% cannot be dealt by this
generator, at any seed, ever.

## Does it matter?

Barely. For the four-league question that started this:

| Event | Probability |
|---|---|
| Some pair of 4 Mode A leagues shares a board **on a given day** | 1.1×10⁻⁷ — 1 in 8.9 million |
| …at least once **over a year** | 4.1×10⁻⁵ — 1 in 24,000 |
| …over ten years | 4.1×10⁻⁴ |

You would need ~24,000 league-years to see it once. Not worth designing around.

The likelier cousin — same root cause — is **one league repeating its own earlier
board**, since that is a birthday problem across a season rather than a handful of
pairs:

| Within | Probability |
|---|---|
| 1 year | 1.2×10⁻³ |
| 5 years | 3.1×10⁻² |
| 10 years | 1.2×10⁻¹ |

~12% over a decade, and only detectable by someone with a long memory.

## A note on the other modes

Mode A is the clean worst case *because* its consumption is a constant 80 draws
per attempt. Modes B and C consume 80 **or 81**, depending on whether the bias
check short-circuits ([`board.ts`](../src/game/board.ts)):

```js
if (n >= 1 && (n >= 6 || rng() < n / 6)) {
```

`n === 0` short-circuits on the left, `n >= 6` on the right — both spend 80 draws.
Only `1 <= n <= 5` spends the extra one. That variable draw breaks the strict
block alignment, so the funnel is weaker there. (Not measured; stated from the
code.)

## Why the obvious fix doesn't work

Hashing or re-mixing the seed before handing it to `mulberry32` **changes
nothing**. Any bijective remap of seed → initial state still lands on the same
single cycle of length 2³², so the effective diversity stays 2³²/43.

The only real fix is a generator with a wider state — e.g. xoshiro128\*\* seeded
via splitmix32 — which would push diversity up against the ~1.66×10¹² board space
instead and drop every number above by roughly 10⁴.

The cost is that stored seeds would stop reproducing their historical boards, so
it is worth checking what replays boards from `daily_seeds` before touching it.
Given the numbers above, there is no urgency.

## Reproducing the prediction

Self-contained, no dependencies — save as `proof.mjs` and run `node proof.mjs`:

```js
const M = 0x6d2b79f5
const DECK = [...Array(81).keys()]
const attr = (c, i) => Math.floor(c / 3 ** i) % 3

function countSets(cards) {
  let n = 0
  for (let i = 0; i < 12; i++)
    for (let j = i + 1; j < 12; j++)
      for (let k = j + 1; k < 12; k++) {
        let ok = true
        for (let x = 0; x < 4; x++)
          if ((attr(cards[i], x) + attr(cards[j], x) + attr(cards[k], x)) % 3) { ok = false; break }
        if (ok) n++
      }
  return n
}

function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a = (a + M) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function modeA(seed) {
  const rng = mulberry32(seed)
  for (let attempts = 1; ; attempts++) {
    const a = DECK.slice()
    for (let i = 80; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] }
    const cards = a.slice(0, 12)
    if (countSets(cards) === 6) return { cards: cards.join(','), attempts }
  }
}

let tested = 0, matched = 0
for (let s = 1000; s < 1400; s++) {
  const base = modeA(s)
  if (base.attempts < 2) continue          // block 1 accepted: the shift skips past it
  if (modeA((s + 80 * M) >>> 0).cards === base.cards) matched++
  tested++
}
console.log(`shift by one 80-draw block: ${matched}/${tested} identical boards`)

let ctrl = 0
for (let s = 1000; s < 1100; s++) if (modeA((s + M) >>> 0).cards === modeA(s).cards) ctrl++
console.log(`control, shift by one draw:  ${ctrl}/100 identical boards`)
```

Expected output:

```
shift by one 80-draw block: 394/394 identical boards
control, shift by one draw:  0/100 identical boards
```
