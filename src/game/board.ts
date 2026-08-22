/**
 * Board generation for the two game modes.
 *
 * Pure and deterministic: every generator takes an explicit RNG, so the same
 * seed always yields the same board. That is what lets slice 3 generate a
 * league board once, server-side, and reproduce or audit it byte-for-byte.
 *
 * The game format is a fixed 12-card puzzle: the board never changes, no
 * replacement, no extra deals. The player's job is to find every set on it.
 */

import { fullDeck, type Card } from './cards'
import { enumerateSets, countSets, type Triple } from './set'

/** A pseudo-random source returning a float in [0, 1), like Math.random. */
export type RNG = () => number

/**
 * The game modes. A league (or practice game) is fixed to one.
 * - A: exactly six sets, count shown.
 * - B: unknown count, ends automatically when the last set is found.
 * - C: unknown count like B, but no auto-end — the player declares "done"
 *   (penalties for declaring early). Generated identically to B.
 */
export type Mode = 'A' | 'B' | 'C'

/** Cards dealt per board. Fixed by the puzzle format. */
export const BOARD_SIZE = 12

/** In Mode A the board contains exactly this many sets, by construction. */
export const MODE_A_SET_COUNT = 6

/**
 * Safety cap on rejection-sampling attempts. Far above anything the maths
 * predicts (Mode A needs on the order of tens); its only job is to turn a
 * hypothetical infinite loop into a loud failure.
 */
const MAX_ATTEMPTS = 100_000

/**
 * A generated board. In league play the whole object is stored server-side and
 * only `cards` is sent to the client — never `sets`, the solution.
 */
export interface Board {
  readonly mode: Mode
  /** The 12 cards, in deal order. This is all the client ever receives. */
  readonly cards: Card[]
  /** The solution: every set on the board, as index triples. Server-only. */
  readonly sets: Triple[]
  /** How many deals rejection sampling took to land this board (>= 1). */
  readonly attempts: number
}

/**
 * mulberry32: a small, fast, seedable PRNG. Deterministic per seed and
 * dependency-free, which is all the board generators need. Not for anything
 * cryptographic.
 */
export function mulberry32(seed: number): RNG {
  let a = seed >>> 0
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Fisher–Yates shuffle. Returns a new array; the input is never mutated, so
 * the shared full deck can be reshuffled endlessly during rejection sampling.
 */
export function shuffle<T>(items: readonly T[], rng: RNG): T[] {
  const a = items.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const tmp = a[i]!
    a[i] = a[j]!
    a[j] = tmp
  }
  return a
}

/** The full deck, built once and reused read-only across every deal. */
const DECK: readonly Card[] = fullDeck()

/** Shuffle the deck and take the first {@link BOARD_SIZE} cards. */
function deal(rng: RNG): Card[] {
  return shuffle(DECK, rng).slice(0, BOARD_SIZE)
}

/**
 * Deal an unknown-count board (Modes B and C), biased toward richer boards: a
 * freshly-dealt board with N sets is kept with probability min(N/6, 1),
 * otherwise re-drawn. Setless boards (N=0) are always rejected — with no auto
 * end they could never finish. This makes 1- and 2-set boards rare so the
 * puzzle stays interesting.
 */
function generateBiasedUnknown(rng: RNG, mode: 'B' | 'C'): Board {
  for (let attempts = 1; attempts <= MAX_ATTEMPTS; attempts++) {
    const cards = deal(rng)
    const n = countSets(cards)
    if (n >= 1 && (n >= 6 || rng() < n / 6)) {
      return { mode, cards, sets: enumerateSets(cards), attempts }
    }
  }
  throw new Error(`generateMode${mode}: no acceptable board within ${MAX_ATTEMPTS} attempts`)
}

/** Mode B — unknown count, auto-ends when the last set is found. */
export function generateModeB(rng: RNG): Board {
  return generateBiasedUnknown(rng, 'B')
}

/**
 * Generate a twelve-card board with no valid SET.
 *
 * This is intentionally separate from the player-facing modes, which always
 * contain at least one answer. Vision-policy evaluation needs both sides of
 * its raw action space (`Set(...)` and `NoSet`) without embedding a fixed
 * corpus in the browser harness.
 */
export function generateNoSetBoard(rng: RNG): Board {
  for (let attempts = 1; attempts <= MAX_ATTEMPTS; attempts++) {
    const cards = deal(rng)
    if (countSets(cards) === 0) {
      return { mode: 'B', cards, sets: [], attempts }
    }
  }
  throw new Error(`generateNoSetBoard: no setless board within ${MAX_ATTEMPTS} attempts`)
}

/**
 * Mode A — exactly six sets. Rejection sampling: reshuffle and re-count until a
 * deal has exactly six sets. A random 12-card board averages ~2.78 sets, so six
 * is above the mean but common enough to hit within tens of attempts. The
 * attempt count is reported on the returned board.
 */
export function generateModeA(rng: RNG): Board {
  for (let attempts = 1; attempts <= MAX_ATTEMPTS; attempts++) {
    const cards = deal(rng)
    if (countSets(cards) === MODE_A_SET_COUNT) {
      return { mode: 'A', cards, sets: enumerateSets(cards), attempts }
    }
  }
  throw new Error(`generateModeA: no ${MODE_A_SET_COUNT}-set board within ${MAX_ATTEMPTS} attempts`)
}

/**
 * Mode C — hardcore. Same biased unknown-count board as Mode B; the difference
 * is entirely in play (no auto-end, declare "done" with penalties).
 */
export function generateModeC(rng: RNG): Board {
  return generateBiasedUnknown(rng, 'C')
}

/** Dispatch to the generator for the given mode. */
export function generateBoard(mode: Mode, rng: RNG): Board {
  if (mode === 'A') return generateModeA(rng)
  if (mode === 'C') return generateModeC(rng)
  return generateModeB(rng)
}
