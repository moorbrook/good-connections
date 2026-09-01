import { useState } from 'react'
import { paceSeries, type PaceSeries } from '../game/pace'
import type { LeaderboardRow } from '../lib/leagues'
import { formatTime } from './format'
import { H, M, OffScaleMark, PLOT_W, W, YAxis, Y_CAP_MS, yScale } from './paceChart'
import { playerColours } from './playerColour'

/**
 * Pace chart for today's identical board: cumulative time (Y) vs sets found
 * (X), one line per player. Because everyone plays the same seed the lines are
 * directly comparable. Only shown once the viewer has completed, so it reveals
 * nothing new.
 *
 * Mode C penalties are drawn as VERTICAL RISERS in the player's own colour,
 * not as free-floating markers. A premature "done" really does add time to that
 * player's clock and to nothing else, so the step is the shape the data already
 * has (see game/pace). It also solves attribution: a riser is part of a line, so
 * it is obvious whose it is. And because only a penalty can advance time without
 * advancing the set count, a strictly vertical segment can mean nothing else.
 *
 * Colours come from the league roster, not today's turnout, so a player keeps
 * their colour on days when someone is absent (see ./playerColour).
 *
 * Box, Y scale and the 6:00 cap come from ./paceChart, shared with the
 * per-player chart.
 */

interface Series {
  userId: string
  name: string
  isYou: boolean
  colour: string
  pace: PaceSeries
}

/** What a tapped element says. Touch has no hover, so detail needs a tap. */
interface Detail {
  name: string
  text: string
  colour: string
}

export function SameBoardCompare({
  rows,
  currentUserId,
  roster,
  historic = false,
}: {
  rows: LeaderboardRow[]
  currentUserId: string
  /** League members in join order; null when the roster could not be read. */
  roster?: string[] | null
  /** A finished day rather than today's live one. */
  historic?: boolean
}) {
  const [detail, setDetail] = useState<Detail | null>(null)

  // On a live day the chart waits until the viewer has finished, so it can
  // never show them how others paced a board they are still playing. On a
  // finished day that is moot — the board is over, and someone who missed it
  // entirely should still be able to look.
  const me = rows.find((r) => r.userId === currentUserId)
  if (rows.length < 2) return null
  if (!historic && (!me || !me.stats.completed)) return null

  const ordered = [...rows].sort((a, b) => a.userId.localeCompare(b.userId))
  const colours = playerColours(
    ordered.map((r) => r.userId),
    roster ?? null,
  )
  const series: Series[] = ordered.map((r) => ({
    userId: r.userId,
    name: r.displayName,
    isYou: r.userId === currentUserId,
    colour: colours.get(r.userId) ?? '#999999',
    pace: paceSeries(r.events),
  }))

  const allVertices = series.flatMap((s) => s.pace.vertices)
  if (allVertices.length === 0) return null

  const scale = yScale(Math.max(...allVertices.map((v) => v.atMs), 1))
  const { y, maxT } = scale

  const maxN = Math.max(...series.map((s) => s.pace.vertices.filter((v) => v.x % 1 === 0).length), 1)
  const xMin = Math.min(1, ...allVertices.map((v) => v.x))
  const xMax = Math.max(maxN, ...allVertices.map((v) => v.x))
  const x = (v: number) =>
    xMax > xMin ? M.l + ((v - xMin) / (xMax - xMin)) * PLOT_W : M.l + PLOT_W / 2

  // Integer set counts only. The old chart also wrote "false done" / "done"
  // under the half-step slots, which is what produced the unreadable
  // "6false done7false done8done" run along the axis. Thin the ticks if a board
  // ever carries enough sets to crowd them.
  // ~25px per label across the 300-unit plot is comfortable, so up to a dozen
  // ticks all fit; only a board with more sets than that needs thinning.
  const tickStep = Math.ceil(maxN / 12)
  const xTicks = Array.from({ length: maxN }, (_, i) => i + 1).filter(
    (n) => n % tickStep === 0 || n === 1 || n === maxN,
  )
  const anyPenalty = series.some((s) => s.pace.risers.length > 0)

  const show = (name: string, colour: string, text: string) => () =>
    setDetail({ name, colour, text })

  return (
    <section className="league-stats">
      <h2 className="section-label">{historic ? 'The board — pace' : 'Today’s board — pace'}</h2>
      <svg
        className="pace-chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Solve pace by player"
      >
        <YAxis scale={scale} />
        {xTicks.map((n) => (
          <text key={n} x={x(n)} y={H - 8} className="pc-xlabel">
            {n}
          </text>
        ))}

        {series.map((s, si) => {
          // Off-scale markers all sit on the top edge, so fan them out sideways
          // by series to keep two slow players from drawing the same triangle.
          const nudge = (si - (series.length - 1) / 2) * 3
          const setPoints = s.pace.vertices.filter((v) => v.x % 1 === 0)
          return (
            <g key={s.userId}>
              {s.pace.vertices.length > 1 && (
                <polyline
                  points={s.pace.vertices.map((v) => `${x(v.x)},${y(v.atMs)}`).join(' ')}
                  fill="none"
                  stroke={s.colour}
                  strokeWidth={s.isYou ? 3 : 2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  opacity={s.isYou ? 1 : 0.85}
                />
              )}

              {/* Penalty risers, emphasised over the polyline in the same
                  colour, with a generous invisible hit area for touch. */}
              {s.pace.risers.map((r, ri) => (
                <g
                  key={`r${ri}`}
                  className="pc-riser"
                  onClick={show(
                    s.name,
                    s.colour,
                    `false “done” at ${formatTime(r.fromMs, true)} · +${formatTime(r.penaltyMs)} penalty`,
                  )}
                >
                  <line
                    x1={x(r.x)}
                    x2={x(r.x)}
                    y1={y(Math.min(r.fromMs, maxT))}
                    y2={y(Math.min(r.toMs, maxT))}
                    stroke={s.colour}
                    strokeWidth={s.isYou ? 5 : 4}
                    strokeLinecap="round"
                    opacity={0.9}
                  />
                  <rect
                    x={x(r.x) - 11}
                    y={y(Math.min(r.toMs, maxT)) - 6}
                    width={22}
                    height={Math.max(y(Math.min(r.fromMs, maxT)) - y(Math.min(r.toMs, maxT)) + 12, 24)}
                    fill="transparent"
                  />
                  <title>
                    {`${s.name} — false “done” at ${formatTime(r.fromMs, true)}, +${formatTime(
                      r.penaltyMs,
                    )}`}
                  </title>
                </g>
              ))}

              {setPoints.map((p) => (
                <g
                  key={p.x}
                  className="pc-hit"
                  onClick={show(s.name, s.colour, `set ${p.x} at ${formatTime(p.atMs, true)}`)}
                >
                  {scale.isOffScale(p.atMs) ? (
                    <OffScaleMark cx={x(p.x) + nudge} cy={y(p.atMs)} colour={s.colour}>
                      <title>{`${s.name} — set ${p.x} at ${formatTime(p.atMs, true)}`}</title>
                    </OffScaleMark>
                  ) : (
                    <circle cx={x(p.x)} cy={y(p.atMs)} r={s.isYou ? 4 : 3} fill={s.colour}>
                      <title>{`${s.name} — set ${p.x} at ${formatTime(p.atMs, true)}`}</title>
                    </circle>
                  )}
                  {/* Touch target: ~44px in CSS pixels once the 360-wide
                      viewBox is scaled to a phone screen. */}
                  <circle cx={x(p.x)} cy={y(p.atMs)} r={11} fill="transparent" />
                </g>
              ))}

              {s.pace.finish && (
                <g
                  className="pc-hit"
                  onClick={show(
                    s.name,
                    s.colour,
                    `finished at ${formatTime(s.pace.finish.atMs, true)}`,
                  )}
                >
                  <rect
                    x={x(s.pace.finish.x) - 4}
                    y={y(Math.min(s.pace.finish.atMs, maxT)) - 4}
                    width={8}
                    height={8}
                    fill={s.colour}
                  >
                    <title>{`${s.name} — finished at ${formatTime(s.pace.finish.atMs, true)}`}</title>
                  </rect>
                  <circle
                    cx={x(s.pace.finish.x)}
                    cy={y(Math.min(s.pace.finish.atMs, maxT))}
                    r={11}
                    fill="transparent"
                  />
                </g>
              )}
            </g>
          )
        })}
      </svg>

      <div className="pace-legend">
        {series.map((s) => (
          <span key={s.userId} className={`pace-key${s.isYou ? ' is-you' : ''}`}>
            <span className="pace-swatch" style={{ background: s.colour }} />
            {s.name}
            {s.isYou ? ' (you)' : ''}
            {s.pace.penaltyMs > 0 && (
              <span className="pace-key-pen"> +{formatTime(s.pace.penaltyMs)}</span>
            )}
          </span>
        ))}
      </div>

      <p className={`pace-detail${detail ? ' is-shown' : ''}`} aria-live="polite">
        {detail ? (
          <>
            <span className="pace-swatch" style={{ background: detail.colour }} />
            <b>{detail.name}</b> — {detail.text}
          </>
        ) : (
          'Tap a point for detail.'
        )}
      </p>

      <p className="muted pace-caption">
        X = sets found, Y = elapsed time (incl. penalties). ▢ done.
        {anyPenalty && ' A vertical step is a penalty for a premature “done”.'}
        {scale.anyOffScale && ` ▲ is past ${formatTime(Y_CAP_MS)}.`}
      </p>
    </section>
  )
}
