import { describe, expect, it } from 'vitest'

import { curriculumVariants } from './render-vision-boards.mjs'

const failure = {
  kind: 'has-set',
  theme: 'classic',
  profile: 'crop',
  viewport: [400, 800],
  dpr: 2,
  boardSeed: 11,
  visualSeed: 22,
  boardIds: Array.from({ length: 12 }, (_, index) => index),
  chosenIds: [0, 1, 2],
}

describe('active curriculum renderer variants', () => {
  it('keeps an exact replay and creates deterministic local and new-board variants', () => {
    const variants = curriculumVariants(failure, 3, 4, 2026090601)
    expect(variants).toHaveLength(4)
    expect(variants[0]).toMatchObject(failure)
    expect(variants[1].boardSeed).toBe(failure.boardSeed)
    expect(variants[1].visualSeed).not.toBe(failure.visualSeed)
    expect(variants[2].boardSeed).toBe(failure.boardSeed)
    expect(variants[2].viewport).not.toEqual(failure.viewport)
    expect(variants[3].boardSeed).not.toBe(failure.boardSeed)
    expect(variants[3].boardIds).toBeUndefined()
    expect(variants[3].chosenIds).toBeUndefined()
    expect(curriculumVariants(failure, 3, 4, 2026090601)).toEqual(variants)
  })
})
