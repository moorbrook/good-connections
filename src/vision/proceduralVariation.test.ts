import { describe, expect, it } from 'vitest'
import { variationFor } from './proceduralVariation'

describe('variationFor', () => {
  it('keeps seed zero as the canonical renderer', () => {
    expect(variationFor(0, 42)).toBeUndefined()
  })

  it('is deterministic and stays within the declared renderer envelope', () => {
    for (let seed = 1; seed <= 32; seed++) {
      for (let cardId = 0; cardId < 81; cardId++) {
        const variation = variationFor(seed, cardId)!
        expect(variationFor(seed, cardId)).toEqual(variation)
        expect(variation.brightness).toBeGreaterThanOrEqual(0.86)
        expect(variation.brightness).toBeLessThan(1.14)
        expect(variation.contrast).toBeGreaterThanOrEqual(0.88)
        expect(variation.contrast).toBeLessThan(1.16)
        expect(variation.hueRotation).toBeGreaterThanOrEqual(-18)
        expect(variation.hueRotation).toBeLessThan(18)
        expect(variation.saturation).toBeGreaterThanOrEqual(0.78)
        expect(variation.saturation).toBeLessThan(1.3)
        expect(variation.spacingScale).toBeGreaterThanOrEqual(0.82)
        expect(variation.spacingScale).toBeLessThan(1.18)
        expect(variation.stripePeriodScale).toBeGreaterThanOrEqual(0.72)
        expect(variation.stripePeriodScale).toBeLessThan(1.28)
        expect(variation.strokeScale).toBeGreaterThanOrEqual(0.76)
        expect(variation.strokeScale).toBeLessThan(1.28)
        expect(variation.symbolRotation).toBeGreaterThanOrEqual(-8)
        expect(variation.symbolRotation).toBeLessThan(8)
        expect(variation.symbolScale).toBeGreaterThanOrEqual(0.82)
        expect(variation.symbolScale).toBeLessThan(1.16)
        expect(variation.symbolY).toBeGreaterThanOrEqual(-5)
        expect(variation.symbolY).toBeLessThan(5)
      }
    }
  })
})
