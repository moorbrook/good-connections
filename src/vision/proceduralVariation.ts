import type { CardVisualVariation } from '../ui/Card'

/** Deterministic renderer variation shared by atlas and responsive test surfaces. */
export function variationFor(seed: number, cardId: number): CardVisualVariation | undefined {
  if (seed === 0) return undefined
  let state = (seed ^ Math.imul(cardId + 1, 0x9e3779b1)) >>> 0
  const random = () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296
  }
  const between = (low: number, high: number) => low + (high - low) * random()
  return {
    brightness: between(0.86, 1.14),
    contrast: between(0.88, 1.16),
    hueRotation: between(-18, 18),
    saturation: between(0.78, 1.3),
    spacingScale: between(0.82, 1.18),
    stripePeriodScale: between(0.72, 1.28),
    strokeScale: between(0.76, 1.28),
    symbolRotation: between(-8, 8),
    symbolScale: between(0.82, 1.16),
    symbolY: between(-5, 5),
  }
}
