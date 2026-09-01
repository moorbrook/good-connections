import { describe, expect, it } from 'vitest'
import { formatClock, formatCountdown, isSlotClosed, nextMidnightMs, zonedDateISO } from './time'

/** Wall-clock reading of an instant in a zone, as "YYYY-MM-DD HH:mm". */
function wallClock(timezone: string, atMs: number): string {
  const p = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone,
    hourCycle: 'h23',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).formatToParts(atMs)
  const get = (t: string) => p.find((x) => x.type === t)!.value
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}`
}

describe('nextMidnightMs', () => {
  it('lands on midnight in the target zone, not the runner’s', () => {
    // 2026-08-18T09:30Z = 17:30 in Singapore.
    const now = Date.parse('2026-08-18T09:30:00Z')
    const next = nextMidnightMs('Asia/Singapore', now)!
    expect(wallClock('Asia/Singapore', next)).toBe('2026-08-19 00:00')
    expect(next).toBeGreaterThan(now)
  })

  it('is strictly in the future even one second before midnight', () => {
    // 15:59:59Z = 23:59:59 in Singapore on the 18th.
    const now = Date.parse('2026-08-18T15:59:59Z')
    const next = nextMidnightMs('Asia/Singapore', now)!
    expect(next - now).toBe(1000)
    expect(wallClock('Asia/Singapore', next)).toBe('2026-08-19 00:00')
  })

  it('rolls over month and year boundaries', () => {
    const next = nextMidnightMs('UTC', Date.parse('2026-12-31T23:00:00Z'))!
    expect(wallClock('UTC', next)).toBe('2027-01-01 00:00')
  })

  it('gets the wall clock right across a DST change', () => {
    // London goes GMT+1 → GMT+0 at 02:00 local on 2026-10-25. Standing just
    // before that on the 24th, the next midnight is 25 hours away, not 24.
    const now = Date.parse('2026-10-24T22:00:00Z') // 23:00 BST on the 24th
    const next = nextMidnightMs('Europe/London', now)!
    expect(wallClock('Europe/London', next)).toBe('2026-10-25 00:00')
    // Midnight is still BST; the change happens at 02:00, later that day.
    expect(next - now).toBe(60 * 60 * 1000)

    // And from inside the 25th, the following midnight is 25 hours out.
    const after = nextMidnightMs('Europe/London', next)!
    expect(wallClock('Europe/London', after)).toBe('2026-10-26 00:00')
    expect(after - next).toBe(25 * 60 * 60 * 1000)
  })

  it('handles a zone west of UTC', () => {
    // 2026-08-18T09:30Z = 05:30 in New York.
    const next = nextMidnightMs('America/New_York', Date.parse('2026-08-18T09:30:00Z'))!
    expect(wallClock('America/New_York', next)).toBe('2026-08-19 00:00')
  })

  it('returns null for an unknown zone instead of throwing', () => {
    expect(nextMidnightMs('Mars/Olympus_Mons', Date.now())).toBeNull()
  })
})

describe('zonedDateISO', () => {
  it('reads the date in the target zone, not the runner’s', () => {
    // 2026-08-18T17:00Z is still the 18th in London but already the 19th in SG.
    const at = Date.parse('2026-08-18T17:00:00Z')
    expect(zonedDateISO('Europe/London', at)).toBe('2026-08-18')
    expect(zonedDateISO('Asia/Singapore', at)).toBe('2026-08-19')
  })

  it('zero-pads month and day', () => {
    expect(zonedDateISO('UTC', Date.parse('2026-01-05T12:00:00Z'))).toBe('2026-01-05')
  })

  it('returns null for an unknown zone', () => {
    expect(zonedDateISO('Mars/Olympus_Mons', Date.now())).toBeNull()
  })
})

describe('isSlotClosed', () => {
  // CORSICA: local midnight in Etc/GMT-1 is 6pm US Eastern Standard Time.
  const tz = 'Etc/GMT-1'

  it('is open while the puzzle’s own day is still running', () => {
    // 2026-01-15T22:00Z is 23:00 in UTC+1 — still the 15th there.
    expect(isSlotClosed(tz, '2026-01-15', Date.parse('2026-01-15T22:00:00Z'))).toBe(false)
  })

  it('closes the moment the league’s day rolls over', () => {
    // 23:00Z = midnight on the 16th in UTC+1, i.e. 6pm EST — the new puzzle.
    expect(isSlotClosed(tz, '2026-01-15', Date.parse('2026-01-15T23:00:00Z'))).toBe(true)
  })

  it('stays closed for older puzzles', () => {
    expect(isSlotClosed(tz, '2026-01-01', Date.parse('2026-01-15T22:00:00Z'))).toBe(true)
  })

  it('treats a future-dated puzzle as open', () => {
    expect(isSlotClosed(tz, '2026-02-01', Date.parse('2026-01-15T22:00:00Z'))).toBe(false)
  })

  it('fails closed for an unknown zone', () => {
    // Anything gated on this reveals a solution, so a bad league row must keep
    // it hidden rather than open it up.
    expect(isSlotClosed('Mars/Olympus_Mons', '2020-01-01', Date.now())).toBe(false)
  })
})

describe('formatCountdown', () => {
  it('shows hours and minutes above an hour', () => {
    expect(formatCountdown(7 * 3600_000 + 12 * 60_000 + 30_000)).toBe('7h 12m')
  })
  it('shows minutes and seconds inside the last hour', () => {
    expect(formatCountdown(12 * 60_000 + 5_000)).toBe('12m 05s')
  })
  it('shows bare seconds inside the last minute', () => {
    expect(formatCountdown(9_000)).toBe('9s')
  })
  it('collapses non-positive to “now”', () => {
    expect(formatCountdown(0)).toBe('now')
    expect(formatCountdown(-5)).toBe('now')
  })
})

describe('formatClock', () => {
  /** An instant at a given time on the runner's own clock, so these hold in any TZ. */
  const localAt = (h: number, m = 0) => {
    const d = new Date()
    d.setHours(h, m, 0, 0)
    return d.getTime()
  }

  it('names the two hours people have words for', () => {
    expect(formatClock(localAt(0))).toBe('midnight')
    expect(formatClock(localAt(12))).toBe('noon')
  })
  it('drops :00 on the hour', () => {
    expect(formatClock(localAt(15))).toBe('3 pm')
    expect(formatClock(localAt(9))).toBe('9 am')
  })
  it('keeps the minutes when there are any', () => {
    expect(formatClock(localAt(15, 30))).toBe('3:30 pm')
    expect(formatClock(localAt(0, 5))).toBe('12:05 am')
    expect(formatClock(localAt(12, 1))).toBe('12:01 pm')
  })
})

describe('a league whose day rolls at 6pm US Eastern Standard Time', () => {
  // CORSICA. Etc/GMT-1 is UTC+1 (POSIX inverts the sign) and never observes
  // DST, so its local midnight is a fixed 23:00 UTC — which is 18:00 EST.
  const inNY = (ms: number) =>
    new Intl.DateTimeFormat('en-CA', {
      timeZone: 'America/New_York',
      hourCycle: 'h23',
      hour: '2-digit',
      minute: '2-digit',
    }).format(ms)

  it('rolls over at 18:00 in New York while Eastern is on standard time', () => {
    expect(inNY(nextMidnightMs('Etc/GMT-1', Date.parse('2026-01-15T00:00:00Z'))!)).toBe('18:00')
  })

  it('is the same instant in July, which Eastern reads as 19:00 on DST', () => {
    // Not a bug: the league is pinned to a fixed offset so the mark stays a
    // stable instant season-round. Eastern is what moves, by an hour, twice a
    // year. Holding 6pm on the Eastern wall clock would need a rollover-hour
    // column in `leagues`, not just a timezone.
    expect(inNY(nextMidnightMs('Etc/GMT-1', Date.parse('2026-07-15T00:00:00Z'))!)).toBe('19:00')
  })

  it('is 23:00 UTC in both January and July', () => {
    const inUTC = (ms: number) =>
      new Intl.DateTimeFormat('en-CA', {
        timeZone: 'UTC',
        hourCycle: 'h23',
        hour: '2-digit',
        minute: '2-digit',
      }).format(ms)
    for (const iso of ['2026-01-15T00:00:00Z', '2026-07-15T00:00:00Z']) {
      expect(inUTC(nextMidnightMs('Etc/GMT-1', Date.parse(iso))!)).toBe('23:00')
    }
  })
})

describe('a league whose day rolls at 3pm Singapore time', () => {
  // Etc/GMT+7 is UTC-7 (POSIX inverts the sign) and never observes DST, so its
  // local midnight is 15:00 in Singapore all year — which is how a non-midnight
  // rollover is configured without a schema change.
  it('rolls over at 15:00 SGT in both January and July', () => {
    const inSGT = (ms: number) =>
      new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Asia/Singapore',
        hourCycle: 'h23',
        hour: '2-digit',
        minute: '2-digit',
      }).format(ms)
    for (const iso of ['2026-01-15T00:00:00Z', '2026-07-15T00:00:00Z']) {
      expect(inSGT(nextMidnightMs('Etc/GMT+7', Date.parse(iso))!)).toBe('15:00')
    }
  })
})
