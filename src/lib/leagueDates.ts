/**
 * Which of a league's past days a member is allowed to look at.
 *
 * Browsing history is safe for any day whose slot has closed — the board is
 * over, so there is nothing left to leak. TODAY is the exception: every member
 * plays the same board, so showing today's leaderboard, board or charts to
 * someone who has not played yet would hand them the answers. That one day is
 * the entire security surface of the date picker, which is why the rule lives
 * in its own pure, tested function rather than inline in a component.
 *
 * ISO dates (YYYY-MM-DD) compare correctly as strings, so no Date objects are
 * needed and no timezone can creep back in here — the caller has already
 * resolved "today" in the league's own zone.
 */
export function viewableDates(
  allDates: readonly string[],
  /** Today's date in the LEAGUE's timezone, or null if the zone is unparseable. */
  todayDate: string | null,
  playedToday: boolean,
): string[] {
  // No "today" means no way to tell which day is still live. Show nothing
  // rather than guess: the cost of being wrong is revealing a live board.
  if (todayDate === null) return []

  return [...new Set(allDates)]
    .filter((d) => {
      if (d > todayDate) return false // future: never, played or not
      if (d === todayDate) return playedToday
      return true
    })
    .sort((a, b) => b.localeCompare(a)) // newest first
}
