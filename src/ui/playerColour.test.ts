import { describe, expect, it } from 'vitest'
import { PLAYER_PALETTE, playerColours } from './playerColour'

describe('playerColours', () => {
  const roster = ['alice', 'bob', 'carol']

  it('assigns palette colours by position in the roster', () => {
    const c = playerColours(roster, roster)
    expect(c.get('alice')).toBe(PLAYER_PALETTE[0])
    expect(c.get('bob')).toBe(PLAYER_PALETTE[1])
    expect(c.get('carol')).toBe(PLAYER_PALETTE[2])
  })

  it('keeps a player’s colour when only some of them played today', () => {
    // The whole point: Telemattic is blue whether or not everyone showed up.
    const everyone = playerColours(roster, roster)
    const justCarol = playerColours(['carol'], roster)
    expect(justCarol.get('carol')).toBe(everyone.get('carol'))
  })

  it('keeps existing colours when a new member joins', () => {
    // league_members orders by joined_at, so a newcomer appends. Nobody shifts.
    const before = playerColours(roster, roster)
    const after = playerColours([...roster, 'dave'], [...roster, 'dave'])
    for (const id of roster) expect(after.get(id)).toBe(before.get(id))
    expect(after.get('dave')).toBe(PLAYER_PALETTE[3])
  })

  it('is independent of the order today’s players arrive in', () => {
    const a = playerColours(['carol', 'alice', 'bob'], roster)
    const b = playerColours(['bob', 'carol', 'alice'], roster)
    for (const id of roster) expect(a.get(id)).toBe(b.get(id))
  })

  it('wraps around when a league outgrows the palette', () => {
    const big = Array.from({ length: PLAYER_PALETTE.length + 2 }, (_, i) => `p${i}`)
    const c = playerColours(big, big)
    expect(c.get(`p${PLAYER_PALETTE.length}`)).toBe(PLAYER_PALETTE[0])
    expect(c.size).toBe(big.length)
  })

  it('still colours a player who has records but has left the league', () => {
    const c = playerColours(['alice', 'ghost'], roster)
    expect(c.get('alice')).toBe(PLAYER_PALETTE[0])
    expect(c.get('ghost')).toBeDefined()
    // and does not steal a current member's colour
    expect(c.get('ghost')).not.toBe(c.get('alice'))
  })

  it('falls back to a stable sort when the roster is unknown', () => {
    // e.g. league_members() has not been run on the database yet.
    const a = playerColours(['carol', 'alice'], null)
    const b = playerColours(['alice', 'carol'], null)
    expect(a.get('alice')).toBe(b.get('alice'))
    expect(a.get('carol')).toBe(b.get('carol'))
    expect(a.get('alice')).not.toBe(a.get('carol'))
  })

  it('handles nobody at all', () => {
    expect(playerColours([], roster).size).toBe(0)
    expect(playerColours([], null).size).toBe(0)
  })
})
