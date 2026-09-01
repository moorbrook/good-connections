import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { SameBoardCompare } from './SameBoardCompare'
import { deriveStats, type GameRecord, type TelemetryEvent } from '../game/telemetry'
import type { LeaderboardRow } from '../lib/leagues'
import { PLAYER_PALETTE } from './playerColour'

/**
 * The chart's Y axis is capped at 6:00 so one very slow game can't flatten
 * everyone else's line. These render it to markup and check the geometry.
 */

const CAP_MS = 6 * 60 * 1000
const PLOT_TOP = 12 // = M.t

function row(userId: string, setTimesMs: number[]): LeaderboardRow {
  const events: TelemetryEvent[] = setTimesMs.map((t, i) => ({
    t_ms: t,
    type: 'set_valid',
    payload: { cards: [0, 1, 2], setIndex: i },
  }))
  events.push({
    t_ms: setTimesMs[setTimesMs.length - 1]! + 1,
    type: 'game_end',
    payload: { reason: 'completed' },
  })
  return {
    userId,
    displayName: userId,
    stats: deriveStats({ events, totalSets: setTimesMs.length } as GameRecord),
    events,
  }
}

/** Every y coordinate in the nth polyline (one per player, in userId order). */
function polylineYs(svg: string, n = 0): number[] {
  const pts = [...svg.matchAll(/<polyline points="([^"]+)"/g)][n]?.[1] ?? ''
  return pts
    .split(' ')
    .filter(Boolean)
    .map((p) => Number(p.split(',')[1]))
}

/** A Mode C row: sets at the given times, with a premature done in between. */
function rowWithFalseDone(
  userId: string,
  setTimesMs: number[],
  falseDone: { atMs: number; penaltyMs: number },
): LeaderboardRow {
  const events: TelemetryEvent[] = []
  let inserted = false
  setTimesMs.forEach((t, i) => {
    if (!inserted && t > falseDone.atMs) {
      events.push({
        t_ms: falseDone.atMs,
        type: 'done_attempt',
        payload: { complete: false, penaltyMs: falseDone.penaltyMs },
      })
      inserted = true
    }
    events.push({ t_ms: t, type: 'set_valid', payload: { cards: [0, 1, 2], setIndex: i } })
  })
  events.push({
    t_ms: setTimesMs[setTimesMs.length - 1]! + 1,
    type: 'game_end',
    payload: { reason: 'completed' },
  })
  return {
    userId,
    displayName: userId,
    stats: deriveStats({ events, totalSets: setTimesMs.length } as GameRecord),
    events,
  }
}

/** Vertical <line> elements (x1 === x2), which is what a penalty riser is. */
function risers(svg: string): { x: number; stroke: string }[] {
  return [...svg.matchAll(/<line ([^>]+)>/g)]
    .map((m) => m[1]!)
    .map((attrs) => ({
      x1: /x1="([-\d.]+)"/.exec(attrs)?.[1],
      x2: /x2="([-\d.]+)"/.exec(attrs)?.[1],
      y1: /y1="([-\d.]+)"/.exec(attrs)?.[1],
      y2: /y2="([-\d.]+)"/.exec(attrs)?.[1],
      stroke: /stroke="([^"]+)"/.exec(attrs)?.[1] ?? '',
    }))
    .filter((l) => l.x1 === l.x2 && l.y1 !== l.y2)
    .map((l) => ({ x: Number(l.x1), stroke: l.stroke }))
}

describe('penalties on the pace chart', () => {
  const rows = [
    rowWithFalseDone('a', [10_000, 60_000, 120_000], { atMs: 30_000, penaltyMs: 5_000 }),
    row('b', [20_000, 70_000, 130_000]),
  ]

  it('draws a vertical riser in the penalised player’s own colour', () => {
    // The old chart drew a fixed pink ring, so you could not tell whose it was.
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={rows} currentUserId="a" roster={['a', 'b']} />,
    )
    const found = risers(svg)
    expect(found).toHaveLength(1)
    expect(found[0]!.stroke).toBe(PLAYER_PALETTE[0]) // a's colour, not a shared pink
  })

  it('gives the riser the height of the penalty', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={rows} currentUserId="a" roster={['a', 'b']} />,
    )
    const line = [...svg.matchAll(/<line ([^>]+)>/g)]
      .map((m) => m[1]!)
      .find((a) => /x1="([-\d.]+)"/.exec(a)?.[1] === /x2="([-\d.]+)"/.exec(a)?.[1])!
    const y1 = Number(/y1="([-\d.]+)"/.exec(line)![1])
    const y2 = Number(/y2="([-\d.]+)"/.exec(line)![1])
    // 5s of a 130s axis over a 170px plot ≈ 6.5px, and up the chart means
    // a SMALLER y, so the top (y2) must be above the foot (y1).
    expect(y1 - y2).toBeCloseTo((5_000 / 130_000) * 170, 0)
  })

  it('no longer writes “false done” along the x axis', () => {
    // This is what produced "6false done7false done8done" in the screenshot.
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={rows} currentUserId="a" roster={['a', 'b']} />,
    )
    expect(svg).not.toContain('false done')
    expect(svg).not.toContain('pc-donelabel')
  })

  it('shows the penalty total beside the player in the legend', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={rows} currentUserId="a" roster={['a', 'b']} />,
    )
    expect(svg).toContain('pace-key-pen')
    expect(svg).toContain('+0:05')
  })

  it('invites a tap rather than a hover', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={rows} currentUserId="a" roster={['a', 'b']} />,
    )
    expect(svg).toContain('Tap a point for detail.')
    expect(svg).not.toContain('Hover a point')
  })
})

describe('player colours', () => {
  const rows = [row('a', [10_000, 60_000]), row('b', [20_000, 70_000])]

  it('takes colour from the roster, not from who played', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={rows} currentUserId="a" roster={['b', 'a']} />,
    )
    // b joined first, so b gets palette[0] even though a sorts first.
    const strokes = [...svg.matchAll(/<polyline [^>]*stroke="([^"]+)"/g)].map((m) => m[1])
    expect(strokes).toEqual([PLAYER_PALETTE[1], PLAYER_PALETTE[0]]) // a, b in userId order
  })

  it('still renders when the roster is unavailable', () => {
    const svg = renderToStaticMarkup(<SameBoardCompare rows={rows} currentUserId="a" />)
    const strokes = [...svg.matchAll(/<polyline [^>]*stroke="([^"]+)"/g)].map((m) => m[1])
    expect(strokes).toEqual([PLAYER_PALETTE[0], PLAYER_PALETTE[1]])
  })
})

describe('SameBoardCompare Y cap', () => {
  it('scales to the data when everyone is inside the cap', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare
        rows={[row('a', [10_000, 60_000, 120_000]), row('b', [20_000, 70_000, 130_000])]}
        currentUserId="a"
      />,
    )
    // Top gridline is labelled with the slowest time, not the cap.
    expect(svg).toContain('>2:10<')
    expect(svg).not.toContain('>6:00<')
    expect(svg).not.toContain('+</text>')
    // No off-scale triangles.
    expect(svg).not.toContain('<path')
  })

  it('clamps a slow player to the top edge and marks the points ▲', () => {
    const slow = 20 * 60 * 1000
    const svg = renderToStaticMarkup(
      <SameBoardCompare
        rows={[row('a', [10_000, 60_000, 120_000]), row('b', [30_000, 400_000, slow])]}
        currentUserId="a"
      />,
    )
    // Axis stops at the cap and says so.
    expect(svg).toContain('6:00+')
    // Two of player b's points are past 6:00 → two triangles.
    expect(svg.match(/<path /g)?.length).toBe(2)
    // The tooltip still carries the true time, not the clamped one.
    expect(svg).toContain('20:00.0')
    // The slow line flattens against the top edge instead of running off it:
    // both off-scale points share the plot-top y.
    const ys = polylineYs(svg, 1)
    expect(ys.slice(1)).toEqual([PLOT_TOP, PLOT_TOP])
  })

  it('never plots above the top edge, whatever the times', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare
        rows={[row('a', [10_000, CAP_MS * 4]), row('b', [30_000, 60_000])]}
        currentUserId="a"
      />,
    )
    for (const m of svg.matchAll(/c?y="([-\d.]+)"/g)) {
      expect(Number(m[1])).toBeGreaterThanOrEqual(0)
    }
  })
})

describe('when the chart is allowed to show', () => {
  const finished = row('a', [10_000, 60_000])
  const other = row('b', [20_000, 70_000])

  it('hides a live day from a viewer who has not finished', () => {
    // They are mid-game on this exact board; other players' pacing would be a
    // hint about a puzzle they are still solving.
    const unfinished = row('a', [10_000])
    unfinished.stats = { ...unfinished.stats, completed: false }
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={[unfinished, other]} currentUserId="a" roster={['a', 'b']} />,
    )
    expect(svg).toBe('')
  })

  it('hides a live day from someone who is not in the leaderboard at all', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={[finished, other]} currentUserId="stranger" roster={['a', 'b']} />,
    )
    expect(svg).toBe('')
  })

  it('shows a finished day even to someone who never played it', () => {
    // The board is over; there is nothing left to give away.
    const svg = renderToStaticMarkup(
      <SameBoardCompare
        rows={[finished, other]}
        currentUserId="stranger"
        roster={['a', 'b']}
        historic
      />,
    )
    expect(svg).toContain('polyline')
    expect(svg).toContain('The board — pace')
    expect(svg).not.toContain('Today’s board')
  })

  it('still needs two players to be worth drawing', () => {
    const svg = renderToStaticMarkup(
      <SameBoardCompare rows={[finished]} currentUserId="a" roster={['a']} historic />,
    )
    expect(svg).toBe('')
  })
})
