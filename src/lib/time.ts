/**
 * Timezone arithmetic for league day boundaries. A league's puzzle rolls over
 * at local midnight in the league's own timezone, so "when is the next puzzle"
 * is the next midnight there — not the viewer's midnight.
 *
 * Pure and dependency-free: everything is derived from Intl.DateTimeFormat,
 * which every target browser has.
 */

interface ZonedParts {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  second: number
}

function zonedParts(timezone: string, atMs: number): ZonedParts {
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    hourCycle: 'h23',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
  const out: Record<string, number> = {}
  for (const p of dtf.formatToParts(atMs)) {
    if (p.type !== 'literal') out[p.type] = Number(p.value)
  }
  return out as unknown as ZonedParts
}

/** Zone offset (wall clock minus UTC, in ms) in effect at a given instant. */
function offsetMsAt(timezone: string, atMs: number): number {
  const p = zonedParts(timezone, atMs)
  const asIfUTC = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second)
  return asIfUTC - Math.floor(atMs / 1000) * 1000
}

/**
 * Epoch ms of the next midnight in `timezone` strictly after `nowMs`.
 * Returns null for an unrecognised timezone rather than throwing, so a bad
 * league row degrades to "no countdown" instead of a blank screen.
 *
 * Two passes over the offset: the first uses the offset in force now, the
 * second the offset in force at the candidate instant, which is what makes it
 * correct across a DST change between now and midnight.
 */
export function nextMidnightMs(timezone: string, nowMs: number): number | null {
  let now: ZonedParts
  try {
    now = zonedParts(timezone, nowMs)
  } catch {
    return null
  }
  // Wall-clock midnight of the day after the zone's current date. Date.UTC
  // rolls day 32 into the next month for us.
  const wall = Date.UTC(now.year, now.month - 1, now.day + 1, 0, 0, 0)
  const first = wall - offsetMsAt(timezone, nowMs)
  return wall - offsetMsAt(timezone, first)
}

/**
 * The current local date in `timezone`, as YYYY-MM-DD. Null for an
 * unrecognised zone, matching {@link nextMidnightMs}.
 *
 * This is the same date `today_puzzle()` computes server-side with
 * `(now() at time zone tz)::date`, so comparing it against a stored
 * `puzzle_date` is how the client tells whether a day's slot has closed.
 */
export function zonedDateISO(timezone: string, atMs: number): string | null {
  try {
    const p = zonedParts(timezone, atMs)
    const pad = (n: number) => n.toString().padStart(2, '0')
    return `${p.year}-${pad(p.month)}-${pad(p.day)}`
  } catch {
    return null
  }
}

/**
 * Has the league's slot for `puzzleDate` closed — i.e. has a later puzzle
 * already started in the league's own timezone?
 *
 * Answers false (the cautious direction) for an unknown zone: anything gated on
 * this reveals a solution, so a bad league row must keep it hidden rather than
 * open it up.
 */
export function isSlotClosed(timezone: string, puzzleDate: string, nowMs: number): boolean {
  const today = zonedDateISO(timezone, nowMs)
  return today === null ? false : today > puzzleDate
}

/**
 * A countdown as coarse text: hours+minutes while there is an hour to go,
 * minutes+seconds inside the last hour. Never shows a bare "0".
 */
export function formatCountdown(ms: number): string {
  if (ms <= 0) return 'now'
  const totalSec = Math.floor(ms / 1000)
  const h = Math.floor(totalSec / 3600)
  const m = Math.floor((totalSec % 3600) / 60)
  const s = totalSec % 60
  if (h > 0) return `${h}h ${m.toString().padStart(2, '0')}m`
  if (m > 0) return `${m}m ${s.toString().padStart(2, '0')}s`
  return `${s}s`
}

/**
 * An instant on the VIEWER's own clock, e.g. "3 pm", "9:30 am", "midnight".
 *
 * Deliberately the viewer's zone and not the league's. A league whose day rolls
 * at 3pm Singapore time is configured as a zone seven hours behind UTC, so
 * naming its zone would tell a Singapore player their puzzle arrives at
 * "midnight GMT-7" — true, and useless. What they want to know is when it lands
 * for them.
 */
export function formatClock(atMs: number): string {
  const d = new Date(atMs)
  const h = d.getHours()
  const m = d.getMinutes()
  if (m === 0 && h === 0) return 'midnight'
  if (m === 0 && h === 12) return 'noon'
  const h12 = h % 12 === 0 ? 12 : h % 12
  const meridiem = h < 12 ? 'am' : 'pm'
  return m === 0 ? `${h12} ${meridiem}` : `${h12}:${m.toString().padStart(2, '0')} ${meridiem}`
}
