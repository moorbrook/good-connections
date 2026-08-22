import { describe, expect, it } from 'vitest'

import { aggregateViewEvidence, pixelGridBoxes } from './multiview.js'

describe('pixel-grid multiview inference', () => {
  it('keeps the historical crop first and bounds every view', () => {
    const boxes = pixelGridBoxes(
      { left: 10.25, top: 4.75, width: 20.5, height: 10.5 },
      2,
      2,
      100,
      50,
    )
    expect(boxes[0]).toEqual({ left: 20, top: 9, width: 41, height: 21 })
    expect(boxes).toHaveLength(4)
    expect(boxes.every((box) => box.left >= 0 && box.top >= 0 && box.width > 0 && box.height > 0)).toBe(true)
  })

  it('averages evidence and selects triples in Rust combination order', () => {
    const attributes = Array.from({ length: 4 * 4 * 3 }, (_, index) => index % 3 === 0 ? 0.8 : 0.1)
    const result = aggregateViewEvidence([
      { tripleScores: [-2, 0.5, 0.1, -1], attributeProbabilities: attributes },
      { tripleScores: [-1, 0.3, 0.7, -1], attributeProbabilities: attributes },
    ], 4)
    expect(result.choice).toEqual([0, 1, 3])
    expect(result.score).toBeCloseTo(0.4)
    expect(result.scoreMargin).toBeCloseTo(0)
    expect(result.scoreMean).toBeCloseTo(-0.425)
    expect(result.positiveScoreFraction).toBe(0.5)
    expect(result.attributeEntropy).toBeGreaterThan(0)
    expect(result.attributeMargin).toBeCloseTo(0.7)
  })
})
