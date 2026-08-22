import { describe, it, expect } from 'vitest'
import { fullDeck, cardId } from './cards'
import { isSet, enumerateSets } from './set'
import {
  mulberry32,
  shuffle,
  generateBoard,
  generateModeA,
  generateModeB,
  generateNoSetBoard,
  BOARD_SIZE,
  MODE_A_SET_COUNT,
  type Board,
} from './board'

describe('mulberry32', () => {
  it('is deterministic for a given seed', () => {
    const a = mulberry32(123)
    const b = mulberry32(123)
    const seqA = Array.from({ length: 10 }, () => a())
    const seqB = Array.from({ length: 10 }, () => b())
    expect(seqA).toEqual(seqB)
  })

  it('produces different streams for different seeds', () => {
    const a = Array.from({ length: 10 }, mulberry32(1))
    const b = Array.from({ length: 10 }, mulberry32(2))
    expect(a).not.toEqual(b)
  })

  it('stays within [0, 1)', () => {
    const rng = mulberry32(99)
    for (let i = 0; i < 1000; i++) {
      const x = rng()
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
  })
})

describe('shuffle', () => {
  it('returns a permutation (same multiset of cards)', () => {
    const deck = fullDeck()
    const shuffled = shuffle(deck, mulberry32(5))
    expect(shuffled).toHaveLength(deck.length)
    const original = deck.map(cardId).sort((a, b) => a - b)
    const after = shuffled.map(cardId).sort((a, b) => a - b)
    expect(after).toEqual(original)
  })

  it('does not mutate the input array', () => {
    const deck = fullDeck()
    const before = deck.map(cardId)
    shuffle(deck, mulberry32(7))
    expect(deck.map(cardId)).toEqual(before)
  })

  it('is deterministic for a given seed', () => {
    const deck = fullDeck()
    const a = shuffle(deck, mulberry32(7)).map(cardId)
    const b = shuffle(deck, mulberry32(7)).map(cardId)
    expect(a).toEqual(b)
  })
})

function assertValidBoard(board: Board): void {
  expect(board.cards).toHaveLength(BOARD_SIZE)
  // all cards distinct
  expect(new Set(board.cards.map(cardId)).size).toBe(BOARD_SIZE)
  // every recorded set is genuinely a set
  for (const [i, j, k] of board.sets) {
    expect(isSet(board.cards[i]!, board.cards[j]!, board.cards[k]!)).toBe(true)
  }
  // recorded solution is exactly the enumeration of the board
  expect(board.sets).toEqual(enumerateSets(board.cards))
}

describe('generateModeB (unknown count)', () => {
  it('produces a valid 12-card board with at least one set', () => {
    const board = generateModeB(mulberry32(1))
    expect(board.mode).toBe('B')
    assertValidBoard(board)
    expect(board.sets.length).toBeGreaterThanOrEqual(1)
  })

  it('never deals a setless board across 500 seeds', () => {
    for (let seed = 0; seed < 500; seed++) {
      const board = generateModeB(mulberry32(seed))
      expect(board.sets.length).toBeGreaterThanOrEqual(1)
    }
  })

  it('is deterministic for a given seed', () => {
    const a = generateModeB(mulberry32(2024))
    const b = generateModeB(mulberry32(2024))
    expect(a.cards.map(cardId)).toEqual(b.cards.map(cardId))
    expect(a.sets).toEqual(b.sets)
  })
})

describe('generateNoSetBoard (vision-policy evaluation)', () => {
  it('produces deterministic valid boards with no SET', () => {
    for (let seed = 0; seed < 200; seed++) {
      const board = generateNoSetBoard(mulberry32(seed))
      assertValidBoard(board)
      expect(board.sets).toHaveLength(0)
    }
    const first = generateNoSetBoard(mulberry32(2026))
    const second = generateNoSetBoard(mulberry32(2026))
    expect(first.cards.map(cardId)).toEqual(second.cards.map(cardId))
  })
})

describe('generateModeA (exactly six sets)', () => {
  it('produces a valid 12-card board with exactly six sets', () => {
    const board = generateModeA(mulberry32(1))
    expect(board.mode).toBe('A')
    assertValidBoard(board)
    expect(board.sets).toHaveLength(MODE_A_SET_COUNT)
  })

  it('always contains exactly six sets across 300 seeds', () => {
    for (let seed = 0; seed < 300; seed++) {
      const board = generateModeA(mulberry32(seed))
      expect(board.sets.length).toBe(MODE_A_SET_COUNT)
    }
  })

  it('is deterministic for a given seed', () => {
    const a = generateModeA(mulberry32(2024))
    const b = generateModeA(mulberry32(2024))
    expect(a.cards.map(cardId)).toEqual(b.cards.map(cardId))
  })

  it('reports how many deals it took (attempts >= 1)', () => {
    const board = generateModeA(mulberry32(1))
    expect(board.attempts).toBeGreaterThanOrEqual(1)
  })

  it('MEASUREMENT: average attempts to hit exactly six sets is modest (tens)', () => {
    let total = 0
    const N = 300
    for (let seed = 0; seed < N; seed++) {
      total += generateModeA(mulberry32(seed)).attempts
    }
    const avg = total / N
    // Reported for visibility; a random 12-card board averages ~2.78 sets,
    // and P(exactly 6) is on the order of a few percent -> tens of attempts.
    // eslint-disable-next-line no-console
    console.log(`Mode A: average ${avg.toFixed(1)} deals per board over ${N} seeds`)
    expect(avg).toBeGreaterThan(1)
    expect(avg).toBeLessThan(200)
  })
})

describe('generateModeC (hardcore)', () => {
  it('produces a Mode C board with at least one set across 200 seeds', () => {
    for (let seed = 0; seed < 200; seed++) {
      const board = generateBoard('C', mulberry32(seed))
      expect(board.mode).toBe('C')
      expect(board.sets.length).toBeGreaterThanOrEqual(1)
      expect(board.sets).toEqual(enumerateSets(board.cards))
    }
  })
})

describe('unknown-count board bias', () => {
  it('keeps richer boards: average set count is well above the unbiased ~2.78', () => {
    let total = 0
    const N = 300
    for (let seed = 0; seed < N; seed++) total += generateBoard('B', mulberry32(seed)).sets.length
    expect(total / N).toBeGreaterThan(3.1)
  })
})

describe('generateBoard dispatch', () => {
  it('delegates to the mode-specific generator', () => {
    expect(generateBoard('A', mulberry32(3)).sets).toHaveLength(MODE_A_SET_COUNT)
    expect(generateBoard('B', mulberry32(3)).sets.length).toBeGreaterThanOrEqual(1)
  })
})
