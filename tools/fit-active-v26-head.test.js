import { describe, expect, it } from 'vitest'

import {
  selectConstrainedCandidate,
  thresholdBoundaries,
} from './fit-active-v26-head.mjs'

function metrics(failures, hasSetFailures, noSetFailures, worstConditional) {
  return { failures, hasSetFailures, noSetFailures, worstConditional }
}

describe('active-v26 recall-constrained threshold selection', () => {
  it('enumerates both constant-action boundaries and every distinct midpoint', () => {
    expect(thresholdBoundaries([3, -1, 1, 1])).toEqual([-2, 0, 2, 6])
  })

  it('rejects a lower-error threshold that violates either conditional limit', () => {
    const selected = selectConstrainedCandidate([
      { threshold: -1, metrics: metrics(4, 1, 3, 0.7) },
      { threshold: 0, metrics: metrics(3, 3, 0, 0.7) },
      { threshold: 1, metrics: metrics(3, 0, 3, 0.7) },
    ], { hasSetFailures: 1, noSetFailures: 3 })
    expect(selected.threshold).toBe(1)
  })

  it('uses the lower threshold as the final recall-favoring tie break', () => {
    const selected = selectConstrainedCandidate([
      { threshold: 0.5, metrics: metrics(2, 1, 1, 0.9) },
      { threshold: -0.5, metrics: metrics(2, 1, 1, 0.9) },
    ], { hasSetFailures: 1, noSetFailures: 1 })
    expect(selected.threshold).toBe(-0.5)
  })
})
