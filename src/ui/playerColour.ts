/**
 * Stable per-player chart colours.
 *
 * A player's colour has to be the same every day, or the pace charts cannot be
 * compared across days — which is the whole reason they exist. Assigning colours
 * from "whoever played today", as the charts used to, means a player's colour
 * moves whenever someone is absent.
 *
 * So colours are assigned from the league's ROSTER, not from today's turnout,
 * and `league_members()` returns that roster in join order. Join order is
 * append-only: a new member takes the next free colour and nobody else moves.
 * (Sorting by user id would have reshuffled everyone who sorts after the
 * newcomer — stable across days, but not across joins.)
 *
 * Okabe-Ito categorical palette, in its published colourblind-safe order.
 */
export const PLAYER_PALETTE = [
  '#56B4E9',
  '#E69F00',
  '#009E73',
  '#F0E442',
  '#CC79A7',
  '#D55E00',
  '#0072B2',
  '#999999',
] as const

/**
 * Map each of `userIds` to its colour.
 *
 * `roster` is the league's members in join order; pass null when it could not be
 * fetched (e.g. the `league_members()` function has not been installed yet), in
 * which case this degrades to a stable sort of the ids present — consistent
 * within a chart, but not guaranteed across days.
 *
 * A player with records who has since left the league keeps a colour too: they
 * are appended after the roster rather than dropped, so their line still draws.
 */
export function playerColours(
  userIds: readonly string[],
  roster: readonly string[] | null,
): Map<string, string> {
  const order = roster ? [...roster] : [...userIds].sort()
  // Anyone with records but no membership (left the league) goes after the
  // roster, in a stable order, so they never displace a current member.
  const known = new Set(order)
  for (const id of [...userIds].sort()) {
    if (!known.has(id)) {
      known.add(id)
      order.push(id)
    }
  }

  const wanted = new Set(userIds)
  const out = new Map<string, string>()
  order.forEach((id, i) => {
    if (wanted.has(id)) out.set(id, PLAYER_PALETTE[i % PLAYER_PALETTE.length]!)
  })
  return out
}
