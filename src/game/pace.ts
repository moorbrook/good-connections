/**
 * Pace-chart geometry and solve-time spread, derived from the event log.
 *
 * Pure and React-free, like the rest of `game/` — the charts import shapes from
 * here and only decide how to paint them.
 *
 * The interesting part is how a Mode C penalty is drawn. The recorder folds
 * penalties into `t_ms` (see Recorder.t), and stamps `done_attempt` with the
 * time BEFORE its own penalty lands — so a premature done is a genuine vertical
 * step in the data: the press happens at `t_ms`, and everything after it is
 * `penaltyMs` higher. Drawing that step as a riser is therefore not decoration,
 * it is the shape the numbers already have. It also attributes the penalty to
 * the line that earned it, which a free-floating marker never could.
 */

import type { TelemetryEvent } from './telemetry'

/** A vertex on the pace polyline. */
export interface PaceVertex {
  /** Sets found at this moment. Done presses sit half a step later, so they
   * never stack on top of a set's own point. */
  x: number
  /** Elapsed ms, including every penalty accrued up to here. */
  atMs: number
}

/** A vertical step in the line where a premature "done" cost time. */
export interface PenaltyRiser {
  x: number
  /** Time at the moment of the press. */
  fromMs: number
  /** `fromMs + penaltyMs` — where the line resumes. */
  toMs: number
  penaltyMs: number
  /** 1-based: the nth premature done of the game (penalties double each time). */
  ordinal: number
}

export interface PaceSeries {
  /** Every vertex in order, risers included — this is the polyline. */
  vertices: PaceVertex[]
  risers: PenaltyRiser[]
  /** The final, correct "done" press (Mode C only), or null. */
  finish: PaceVertex | null
  /** Total penalty time charged across the game. */
  penaltyMs: number
}

/**
 * Turn one game's log into the pace polyline.
 *
 * A done press is placed at `setsFound + 0.5` so it occupies its own slot
 * between two sets rather than colliding with a set's marker. Two premature
 * dones at the same set count therefore stack on one x, which reads correctly:
 * a single, taller jump.
 */
export function paceSeries(events: readonly TelemetryEvent[]): PaceSeries {
  const vertices: PaceVertex[] = []
  const risers: PenaltyRiser[] = []
  let finish: PaceVertex | null = null
  let setsFound = 0
  let penaltyMs = 0
  let ordinal = 0

  for (const ev of events) {
    if (ev.type === 'set_valid') {
      setsFound++
      vertices.push({ x: setsFound, atMs: ev.t_ms })
    } else if (ev.type === 'done_attempt') {
      const x = setsFound + 0.5
      if (ev.payload.complete) {
        finish = { x, atMs: ev.t_ms }
        vertices.push(finish)
      } else if (ev.payload.penaltyMs > 0) {
        const toMs = ev.t_ms + ev.payload.penaltyMs
        risers.push({
          x,
          fromMs: ev.t_ms,
          toMs,
          penaltyMs: ev.payload.penaltyMs,
          ordinal: ++ordinal,
        })
        penaltyMs += ev.payload.penaltyMs
        // Both ends on the line, so the polyline actually draws the step
        // instead of absorbing it into the slope of the next segment.
        vertices.push({ x, atMs: ev.t_ms }, { x, atMs: toMs })
      }
    }
  }

  return { vertices, risers, finish, penaltyMs }
}

/** Best/worst/mean/spread over a set of solve times. */
export interface TimeSpread {
  bestMs: number
  worstMs: number
  meanMs: number
  /** Population standard deviation (see note in {@link timeSpread}). */
  stdDevMs: number
  count: number
}

/**
 * Summarise a player's solve times.
 *
 * Population standard deviation, not the sample (n-1) one: these are all the
 * games the player has actually played, not a sample drawn from some larger
 * pool we are trying to infer. It also means a single game reports a spread of
 * 0 rather than NaN, which is both true and displayable.
 *
 * Feed this COMPLETED games only. An abandoned game has a shorter elapsed time
 * than a finished one, so including them would make giving up look fast.
 */
export function timeSpread(timesMs: readonly number[]): TimeSpread | null {
  if (timesMs.length === 0) return null
  const n = timesMs.length
  const mean = timesMs.reduce((a, b) => a + b, 0) / n
  const variance = timesMs.reduce((acc, t) => acc + (t - mean) ** 2, 0) / n
  return {
    bestMs: Math.min(...timesMs),
    worstMs: Math.max(...timesMs),
    meanMs: mean,
    stdDevMs: Math.sqrt(variance),
    count: n,
  }
}
