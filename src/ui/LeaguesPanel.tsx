import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { getMyLeagues, getPlayedToday, joinLeague, type League } from '../lib/leagues'

interface LeaguesPanelProps {
  onSelectLeague: (league: League, intent?: 'play' | 'standings') => void
}

export function LeaguesPanel({ onSelectLeague }: LeaguesPanelProps) {
  const [leagues, setLeagues] = useState<League[] | null>(null)
  const [played, setPlayed] = useState<Record<string, boolean>>({})
  const [code, setCode] = useState('')
  const [joining, setJoining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    getMyLeagues()
      .then((ls) => {
        setLeagues(ls)
        // Which of them have a game waiting. A failure here just means every
        // card renders in its neutral state, so it never touches `error`.
        return getPlayedToday(ls).then(setPlayed)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const join = async (e: FormEvent) => {
    e.preventDefault()
    const value = code.trim()
    if (!value) return
    setJoining(true)
    setError(null)
    try {
      await joinLeague(value)
      setCode('')
      load()
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(
        /no league with that code/i.test(msg)
          ? 'No league found for that code. Check it and try again.'
          : msg,
      )
    } finally {
      setJoining(false)
    }
  }

  return (
    <section className="leagues-section">
      <h2 className="section-label">Leagues</h2>

      {leagues && leagues.length > 0 && (
        <div className="league-list">
          {leagues.map((l) => {
            const done = played[l.id] === true
            return (
              <div key={l.id} className={`league-card ${done ? 'is-played' : 'is-fresh'}`}>
                <button
                  type="button"
                  className="league-card-main"
                  onClick={() => onSelectLeague(l, 'play')}
                >
                  <span className="league-name">
                    {l.name}
                    {/* A dot as well as the colour: the card colours already
                        carry meaning elsewhere in this app, and colour alone is
                        a poor sole signal. */}
                    {!done && <span className="league-dot" aria-hidden="true" />}
                  </span>
                  <span className="league-meta">
                    Mode {l.mode} ·{' '}
                    {done ? 'played today · see the results →' : 'new puzzle ready →'}
                  </span>
                </button>
                {/* Only when unplayed. Once you've played, the main tap already
                    lands on the results page, so a second route would be noise.
                    LeagueResult hides today from the picker on this path, so it
                    cannot be used to scout a board before playing it. */}
                {!done && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm league-card-alt"
                    onClick={() => onSelectLeague(l, 'standings')}
                  >
                    Past results
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
      {leagues && leagues.length === 0 && (
        <p className="muted">You haven’t joined a league yet. Enter a code below.</p>
      )}

      <form className="join-form" onSubmit={join}>
        <input
          className="auth-input"
          placeholder="League code (e.g. SG-TEST)"
          value={code}
          onChange={(e) => setCode(e.target.value)}
        />
        <button type="submit" className="btn btn-ghost btn-sm" disabled={joining || !code.trim()}>
          {joining ? 'Joining…' : 'Join'}
        </button>
      </form>
      {error && <p className="auth-error">{error}</p>}
    </section>
  )
}
