import type { Board } from '../game/board'
import { matchHighlights } from '../game/telemetry'
import type { LeaderboardRow } from '../lib/leagues'
import { Card } from './Card'
import { formatTime, setLabel } from './format'

/**
 * The day's board, read through the players: for each of them the set that cost
 * them the most (their longest stall) and the set they found last. Everyone
 * plays the identical board, so when the same letter keeps appearing you are
 * looking at a genuinely nasty set rather than one person having a bad day.
 *
 * Sets are named by their position in the solution and never by their cards, so
 * this is safe to show to everyone who has played — including someone who gave
 * up, who must not be handed a solution they could pass on.
 */

interface Entry {
  userId: string
  name: string
  isYou: boolean
  hardest: { label: string; gapMs: number } | null
  last: { label: string; atMs: number } | null
}

/** The set index most players share, if enough of them do to be worth saying. */
function consensus(indices: number[]): { label: string; count: number } | null {
  if (indices.length < 2) return null
  const counts = new Map<number, number>()
  for (const i of indices) counts.set(i, (counts.get(i) ?? 0) + 1)
  let top: [number, number] | null = null
  for (const entry of counts) {
    if (top === null || entry[1] > top[1]) top = entry
  }
  return top !== null && top[1] >= 2 ? { label: setLabel(top[0]), count: top[1] } : null
}

/**
 * Whether it is safe to show the day's actual sets.
 *
 * Two ways in, and both are genuinely closed:
 *   - every member has played, so there is nobody left to leak to; or
 *   - the day's slot has rolled over, so the board is history.
 *
 * The second is what stops a day staying locked forever because one member was
 * away — the first alone would do that. If the roster could not be read the
 * "everyone played" test is unavailable, and this falls back to waiting for the
 * slot to close, which is the cautious direction.
 */
export type RevealReason = 'all-played' | 'slot-closed'

export function canRevealSets(args: {
  playedUserIds: readonly string[]
  roster: readonly string[] | null
  slotClosed: boolean
}): RevealReason | null {
  // "Everyone played" is checked first so the caption can say the true reason
  // on a day that satisfies both.
  if (args.roster && args.roster.length > 0) {
    const played = new Set(args.playedUserIds)
    if (args.roster.every((id) => played.has(id))) return 'all-played'
  }
  return args.slotClosed ? 'slot-closed' : null
}

export function MatchSummary({
  rows,
  currentUserId,
  board,
  reveal = null,
  historic = false,
}: {
  rows: LeaderboardRow[]
  currentUserId: string
  /** A finished day rather than today's live one. */
  historic?: boolean
  /** Today's board. Only needed to draw the sets once they may be revealed. */
  board?: Board | null
  /** Caller's verdict from {@link canRevealSets}; null keeps the cards hidden. */
  reveal?: RevealReason | null
}) {
  const played = rows.map((r) => ({ row: r, high: matchHighlights(r.events) }))
  const entries: Entry[] = played.map(({ row, high }) => ({
    userId: row.userId,
    name: row.displayName,
    isYou: row.userId === currentUserId,
    hardest: high.hardest && {
      label: setLabel(high.hardest.setIndex),
      gapMs: high.hardest.sincePrevMs,
    },
    last: high.last && { label: setLabel(high.last.setIndex), atMs: high.last.atMs },
  }))
  if (entries.every((e) => e.last === null)) return null

  const wall = consensus(played.flatMap((p) => (p.high.hardest ? [p.high.hardest.setIndex] : [])))
  // "Last to fall" only means something for a game that finished: where someone
  // who gave up happened to stop is not the board's final set.
  const finishers = played.filter((p) => p.row.stats.completed)
  const lastToFall = consensus(finishers.flatMap((p) => (p.high.last ? [p.high.last.setIndex] : [])))
  const players = entries.length

  return (
    <section className="league-stats">
      <h2 className="section-label">{historic ? 'The match' : 'Today’s match'}</h2>

      {(wall || lastToFall) && (
        <p className="match-note">
          {wall && (
            <>
              Set <b>{wall.label}</b> was the wall — the longest stall for {wall.count} of {players}.
            </>
          )}
          {wall && lastToFall && ' '}
          {lastToFall && (
            <>
              Set <b>{lastToFall.label}</b> was the last to fall for {lastToFall.count} of{' '}
              {finishers.length} finishers.
            </>
          )}
        </p>
      )}

      <ol className="member-list">
        {entries.map((e) => (
          <li key={e.userId} className={`member-row${e.isYou ? ' is-you' : ''}`}>
            <span className="member-name">
              {e.name}
              {e.isYou ? ' (you)' : ''}
            </span>
            <span className="member-detail">
              {e.last === null ? (
                'no sets found'
              ) : (
                <>
                  {e.hardest && (
                    <>
                      hardest <b>{e.hardest.label}</b> +{formatTime(e.hardest.gapMs)} ·{' '}
                    </>
                  )}
                  last <b>{e.last.label}</b> at {formatTime(e.last.atMs)}
                </>
              )}
            </span>
          </li>
        ))}
      </ol>

      {reveal && board && (
        <div className="set-reveal">
          {board.sets.map((set, si) => (
            <div className="set-reveal-item" key={si}>
              <span className="set-reveal-label">{setLabel(si)}</span>
              <div className="missed-set">
                {set.map((ci) => (
                  <div className="mini-card" key={ci}>
                    <Card card={board.cards[ci]!} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="muted timeline-hint">
        Sets are lettered by their place in the board’s solution — the same letter is the same set
        for everyone.
        {/* Three states, not two: the gate can be open while the board is
            still unavailable (a finished day whose seed cannot be fetched),
            and saying "hidden while the day is live" there would be a lie. */}
        {!reveal
          ? ' The cards stay hidden while the day is live.'
          : board
            ? reveal === 'all-played'
              ? ' Everyone has played, so here they are.'
              : ' The day is over, so here they are.'
            : ''}
      </p>
    </section>
  )
}
