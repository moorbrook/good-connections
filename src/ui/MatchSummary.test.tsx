import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { canRevealSets, MatchSummary } from './MatchSummary'
import { deriveStats, type GameRecord, type TelemetryEvent } from '../game/telemetry'
import type { LeaderboardRow } from '../lib/leagues'

/** A player's day: [t_ms, setIndex] pairs, in the order they found them. */
function row(name: string, finds: [number, number][]): LeaderboardRow {
  const events: TelemetryEvent[] = finds.map(([t, setIndex]) => ({
    t_ms: t,
    type: 'set_valid',
    payload: { cards: [0, 1, 2], setIndex },
  }))
  events.push({ t_ms: (finds.at(-1)?.[0] ?? 0) + 1_000, type: 'game_end', payload: { reason: 'completed' } })
  return {
    userId: name,
    displayName: name,
    stats: deriveStats({ events } as GameRecord),
    events,
  }
}

const text = (svgOrHtml: string) => svgOrHtml.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ')

describe('MatchSummary', () => {
  it('names each player’s hardest and last set by board letter', () => {
    const html = renderToStaticMarkup(
      <MatchSummary
        rows={[row('aoife', [[10_000, 0], [30_000, 2], [150_000, 4]])]}
        currentUserId="aoife"
      />,
    )
    // The 120s gap before set index 4 (= "E") is the wall; E is also last.
    expect(text(html)).toContain('hardest E +2:00 · last E at 2:30')
  })

  it('calls out the set that stalled most players', () => {
    const html = renderToStaticMarkup(
      <MatchSummary
        rows={[
          row('aoife', [[10_000, 0], [20_000, 1], [140_000, 3]]),
          row('ben', [[15_000, 1], [25_000, 0], [180_000, 3]]),
          row('cleo', [[12_000, 0], [130_000, 3], [140_000, 1]]),
        ]}
        currentUserId="aoife"
      />,
    )
    expect(text(html)).toContain('Set D was the wall — the longest stall for 3 of 3')
    expect(text(html)).toContain('Set D was the last to fall for 2 of 3 finishers')
  })

  it('leaves an abandoned game out of “last to fall” — where they stopped is not the end', () => {
    const quit = row('cleo', [[10_000, 3], [20_000, 3]])
    quit.events.at(-1)!.payload = { reason: 'abandoned' }
    quit.stats = { ...quit.stats, completed: false, abandoned: true }
    const html = renderToStaticMarkup(
      <MatchSummary
        rows={[row('aoife', [[10_000, 0], [20_000, 5]]), row('ben', [[10_000, 0], [20_000, 1]]), quit]}
        currentUserId="aoife"
      />,
    )
    // Only aoife and ben finished, and their last sets differ — no consensus.
    expect(html).not.toContain('last to fall')
    // Cleo still gets her own row.
    expect(text(html)).toContain('cleo')
  })

  it('says nothing about consensus when there is none', () => {
    const html = renderToStaticMarkup(
      <MatchSummary
        rows={[
          row('aoife', [[10_000, 0], [140_000, 1]]),
          row('ben', [[10_000, 2], [140_000, 3]]),
        ]}
        currentUserId="aoife"
      />,
    )
    expect(html).not.toContain('was the wall')
    expect(html).not.toContain('last to fall')
  })

  it('identifies sets by letter and never by their cards', () => {
    const withCards = row('aoife', [[10_000, 0], [90_000, 5]])
    // Distinctive board positions, so a leak would be unmistakable.
    for (const ev of withCards.events) {
      if (ev.type === 'set_valid') ev.payload.cards = [7, 9, 11]
    }
    const body = text(renderToStaticMarkup(<MatchSummary rows={[withCards]} currentUserId="aoife" />))
    expect(body).toContain('hardest F')
    for (const card of ['7', '9', '11']) expect(body).not.toContain(card)
  })

  it('handles a player who found one set, or none', () => {
    const html = renderToStaticMarkup(
      <MatchSummary
        rows={[row('aoife', [[10_000, 3]]), row('ben', [])]}
        currentUserId="aoife"
      />,
    )
    expect(text(html)).toContain('last D at 0:10')
    expect(text(html)).not.toContain('hardest')
    expect(text(html)).toContain('no sets found')
  })

  it('renders nothing when nobody found anything', () => {
    expect(renderToStaticMarkup(<MatchSummary rows={[row('aoife', [])]} currentUserId="aoife" />)).toBe(
      '',
    )
  })
})

/**
 * The reveal gate. Everyone in a league plays the same board, so showing the
 * solution to someone while another member still has the day ahead of them
 * hands over something they can pass on. These are the cases that must stay
 * shut.
 */
describe('canRevealSets', () => {
  const roster = ['alice', 'bob', 'carol']

  it('stays shut while a member has not played', () => {
    expect(
      canRevealSets({ playedUserIds: ['alice', 'bob'], roster, slotClosed: false }),
    ).toBeNull()
  })

  it('opens once every member has played, even mid-day', () => {
    expect(
      canRevealSets({ playedUserIds: ['alice', 'bob', 'carol'], roster, slotClosed: false }),
    ).toBe('all-played')
  })

  it('opens when the day rolls over, even if someone never played', () => {
    // Otherwise one absent member locks that day's board forever.
    expect(canRevealSets({ playedUserIds: ['alice'], roster, slotClosed: true })).toBe(
      'slot-closed',
    )
  })

  it('names "all played" when both gates are open, since that is the real reason', () => {
    expect(
      canRevealSets({ playedUserIds: roster, roster, slotClosed: true }),
    ).toBe('all-played')
  })

  it('stays shut when the roster is unknown and the day is live', () => {
    // league_members() not installed: the "everyone played" test is
    // unavailable, so fall back to waiting for the slot. Failing open here
    // would leak the board to whoever finished first.
    expect(canRevealSets({ playedUserIds: ['alice'], roster: null, slotClosed: false })).toBeNull()
  })

  it('still opens on slot close when the roster is unknown', () => {
    expect(canRevealSets({ playedUserIds: ['alice'], roster: null, slotClosed: true })).toBe(
      'slot-closed',
    )
  })

  it('treats an empty roster as unknown rather than “everyone has played”', () => {
    // [] would vacuously satisfy every(), which would open the gate on day one
    // of a league before anyone joined. That must not count.
    expect(canRevealSets({ playedUserIds: [], roster: [], slotClosed: false })).toBeNull()
  })

  it('is not fooled by a non-member having played', () => {
    expect(
      canRevealSets({
        playedUserIds: ['alice', 'bob', 'stranger'],
        roster,
        slotClosed: false,
      }),
    ).toBeNull()
  })
})

describe('MatchSummary set reveal', () => {
  const rows = [row('MDS', [[1_000, 0]]), row('Telemattic', [[2_000, 1]])]
  const board = {
    mode: 'A' as const,
    cards: Array.from({ length: 12 }, (_, i) => ({
      count: (i % 3) + 1,
      colour: ['red', 'green', 'purple'][i % 3],
      shape: ['diamond', 'squiggle', 'oval'][i % 3],
      fill: ['solid', 'striped', 'open'][i % 3],
      id: i,
    })),
    sets: [
      [0, 1, 2],
      [3, 4, 5],
    ],
    attempts: 1,
  }

  it('draws no cards while the day is live', () => {
    const svg = renderToStaticMarkup(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <MatchSummary rows={rows} currentUserId="MDS" board={board as any} reveal={null} />,
    )
    expect(svg).not.toContain('set-reveal')
    expect(svg).toContain('stay hidden while the day is live')
  })

  it('draws the lettered sets once the gate opens', () => {
    const svg = renderToStaticMarkup(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <MatchSummary rows={rows} currentUserId="MDS" board={board as any} reveal="all-played" />,
    )
    expect(svg).toContain('set-reveal')
    expect(svg).toContain('Everyone has played')
    // Two sets, three cards each.
    expect(svg.match(/mini-card/g)?.length).toBe(6)
  })

  it('says the day is over when that was the reason', () => {
    const svg = renderToStaticMarkup(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <MatchSummary rows={rows} currentUserId="MDS" board={board as any} reveal="slot-closed" />,
    )
    expect(svg).toContain('The day is over')
  })

  it('does not break when the gate is open but the board has not loaded', () => {
    const svg = renderToStaticMarkup(
      <MatchSummary rows={rows} currentUserId="MDS" board={null} reveal="all-played" />,
    )
    expect(svg).not.toContain('set-reveal')
  })
})
