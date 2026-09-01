import { describe, expect, it } from 'vitest'
import { paceSeries, timeSpread } from './pace'
import type { TelemetryEvent } from './telemetry'

/** A set_valid at t, found in solution slot `idx`. */
const set = (t: number, idx = 0): TelemetryEvent => ({
  t_ms: t,
  type: 'set_valid',
  payload: { cards: [0, 1, 2], setIndex: idx },
})
/** A done press. `penalty` > 0 implies a premature one. */
const done = (t: number, complete: boolean, penalty: number): TelemetryEvent => ({
  t_ms: t,
  type: 'done_attempt',
  payload: { complete, penaltyMs: penalty },
})

describe('paceSeries', () => {
  it('places one vertex per found set, at the set count', () => {
    const s = paceSeries([set(1000), set(2500), set(4000)])
    expect(s.vertices).toEqual([
      { x: 1, atMs: 1000 },
      { x: 2, atMs: 2500 },
      { x: 3, atMs: 4000 },
    ])
    expect(s.risers).toEqual([])
    expect(s.finish).toBeNull()
  })

  it('is empty for a log with no sets', () => {
    const s = paceSeries([])
    expect(s.vertices).toEqual([])
    expect(s.risers).toEqual([])
  })

  describe('a premature done', () => {
    // The recorder stamps done_attempt with the time BEFORE its own penalty is
    // applied, then adds the penalty to every later event. So the riser runs
    // from the event's own t_ms up by exactly penaltyMs, and the next set's
    // t_ms already sits above it. See useSetGame.pressDone.
    const events = [set(1000), done(2000, false, 5000), set(9000)]

    it('rises vertically by the penalty, half a step after the set count', () => {
      const s = paceSeries(events)
      expect(s.risers).toEqual([
        { x: 1.5, fromMs: 2000, toMs: 7000, penaltyMs: 5000, ordinal: 1 },
      ])
    })

    it('puts both ends of the riser on the line, so the jump is drawn', () => {
      const s = paceSeries(events)
      expect(s.vertices).toEqual([
        { x: 1, atMs: 1000 },
        { x: 1.5, atMs: 2000 }, // foot of the riser
        { x: 1.5, atMs: 7000 }, // top, +5s
        { x: 2, atMs: 9000 },
      ])
    })

    it('leaves the following set above the riser, not below it', () => {
      const s = paceSeries(events)
      const top = s.risers[0]!.toMs
      const nextSet = s.vertices.at(-1)!
      expect(nextSet.atMs).toBeGreaterThanOrEqual(top)
    })
  })

  it('stacks two false dones at the same set count into one taller jump', () => {
    // Second penalty doubles, and its t_ms already includes the first.
    const s = paceSeries([set(1000), done(2000, false, 5000), done(8000, false, 10000)])
    expect(s.risers).toEqual([
      { x: 1.5, fromMs: 2000, toMs: 7000, penaltyMs: 5000, ordinal: 1 },
      { x: 1.5, fromMs: 8000, toMs: 18000, penaltyMs: 10000, ordinal: 2 },
    ])
    // Monotonic up the same x: the line reads as one tall riser.
    expect(s.vertices.filter((v) => v.x === 1.5).map((v) => v.atMs)).toEqual([
      2000, 7000, 8000, 18000,
    ])
  })

  it('handles a false done before any set is found', () => {
    const s = paceSeries([done(500, false, 5000), set(7000)])
    expect(s.risers[0]).toEqual({ x: 0.5, fromMs: 500, toMs: 5500, penaltyMs: 5000, ordinal: 1 })
  })

  it('records the winning done as the finish, not a riser', () => {
    const s = paceSeries([set(1000), set(2000), done(2600, true, 0)])
    expect(s.risers).toEqual([])
    expect(s.finish).toEqual({ x: 2.5, atMs: 2600 })
    expect(s.vertices.at(-1)).toEqual({ x: 2.5, atMs: 2600 })
  })

  it('ignores events that are not sets or done presses', () => {
    const noise: TelemetryEvent[] = [
      { t_ms: 10, type: 'card_select', payload: { card: 3 } },
      { t_ms: 20, type: 'set_invalid', payload: { cards: [0, 1, 2] } },
      { t_ms: 30, type: 'abandon_prompt', payload: {} },
    ]
    expect(paceSeries([...noise, set(1000)]).vertices).toEqual([{ x: 1, atMs: 1000 }])
  })

  it('totals the penalty time it charged', () => {
    expect(paceSeries([set(1000), done(2000, false, 5000)]).penaltyMs).toBe(5000)
    expect(paceSeries([set(1000)]).penaltyMs).toBe(0)
  })
})

describe('timeSpread', () => {
  it('is null with nothing to summarise', () => {
    expect(timeSpread([])).toBeNull()
  })

  it('reports best, worst and mean', () => {
    const s = timeSpread([30_000, 10_000, 20_000])!
    expect(s.bestMs).toBe(10_000)
    expect(s.worstMs).toBe(30_000)
    expect(s.meanMs).toBe(20_000)
    expect(s.count).toBe(3)
  })

  it('uses the population standard deviation', () => {
    // mean 20s; deviations -10, 0, +10 => variance 200/3 => sd 8.165s
    const s = timeSpread([10_000, 20_000, 30_000])!
    expect(s.stdDevMs).toBeCloseTo(8164.97, 0)
  })

  it('gives a single game a spread of zero rather than NaN', () => {
    // The sample (n-1) formula divides by zero here. Population does not.
    const s = timeSpread([42_000])!
    expect(s.stdDevMs).toBe(0)
    expect(s.bestMs).toBe(42_000)
    expect(s.worstMs).toBe(42_000)
    expect(s.meanMs).toBe(42_000)
  })

  it('is zero for a perfectly consistent player', () => {
    expect(timeSpread([15_000, 15_000, 15_000])!.stdDevMs).toBe(0)
  })

  it('does not mutate the array it was handed', () => {
    const input = [30_000, 10_000, 20_000]
    timeSpread(input)
    expect(input).toEqual([30_000, 10_000, 20_000])
  })
})
