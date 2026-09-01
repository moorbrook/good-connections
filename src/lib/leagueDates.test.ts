import { describe, expect, it } from 'vitest'
import { viewableDates } from './leagueDates'

describe('viewableDates', () => {
  const history = ['2026-08-18', '2026-08-19', '2026-08-20']

  it('lists past days newest first', () => {
    expect(viewableDates(history, '2026-08-21', false)).toEqual([
      '2026-08-20',
      '2026-08-19',
      '2026-08-18',
    ])
  })

  it('hides today until you have played it', () => {
    // Otherwise the date picker hands you today's board and everyone's times
    // before you have played — the whole comparison would be void.
    expect(viewableDates(history, '2026-08-20', false)).toEqual(['2026-08-19', '2026-08-18'])
  })

  it('shows today once you have played it', () => {
    expect(viewableDates(history, '2026-08-20', true)).toEqual([
      '2026-08-20',
      '2026-08-19',
      '2026-08-18',
    ])
  })

  it('never shows a date after today, played or not', () => {
    // Defensive: a clock skew or a bad row must not expose an unplayed board.
    const withFuture = [...history, '2026-08-25']
    expect(viewableDates(withFuture, '2026-08-20', true)).not.toContain('2026-08-25')
    expect(viewableDates(withFuture, '2026-08-20', false)).not.toContain('2026-08-25')
  })

  it('de-duplicates repeated dates', () => {
    expect(viewableDates(['2026-08-19', '2026-08-19'], '2026-08-20', false)).toEqual(['2026-08-19'])
  })

  it('shows nothing when the only day on record is an unplayed today', () => {
    expect(viewableDates(['2026-08-20'], '2026-08-20', false)).toEqual([])
  })

  it('hides everything when the league-local date is unknown', () => {
    // A timezone the browser cannot parse. Without a "today" there is no way to
    // tell which day is still live, so showing nothing is the safe answer.
    expect(viewableDates(history, null, false)).toEqual([])
    expect(viewableDates(history, null, true)).toEqual([])
  })

  it('copes with no history at all', () => {
    expect(viewableDates([], '2026-08-20', true)).toEqual([])
  })

  it('does not mutate its input', () => {
    const input = ['2026-08-18', '2026-08-20', '2026-08-19']
    viewableDates(input, '2026-08-21', false)
    expect(input).toEqual(['2026-08-18', '2026-08-20', '2026-08-19'])
  })
})
