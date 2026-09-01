import { useMemo } from 'react'
import type { Mode } from '../game/board'
import { aggregateSolveTimes, completionRate } from '../game/telemetry'
import { loadRecords } from './storage'
import { formatTime } from './format'
import { Card } from './Card'
import { CARD_THEMES, type CardTheme } from './cardThemes'
import type { AuthState } from './useAuth'
import { LeaguesPanel } from './LeaguesPanel'
import type { League } from '../lib/leagues'

interface MenuProps {
  onStart: (mode: Mode) => void
  /** Bumped after each game so the history re-reads from storage. */
  refreshKey: number
  cardTheme: CardTheme
  onCardThemeChange: (theme: CardTheme) => void
  auth: AuthState
  onSignIn: () => void
  onSignOut: () => void
  onSelectLeague: (league: League, intent?: 'play' | 'standings') => void
}

function AuthBar({
  auth,
  onSignIn,
  onSignOut,
}: {
  auth: AuthState
  onSignIn: () => void
  onSignOut: () => void
}) {
  if (auth.status === 'signedIn') {
    return (
      <div className="auth-bar">
        <span className="auth-who">
          Signed in as <strong>{auth.displayName}</strong>
        </span>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onSignOut}>
          Sign out
        </button>
      </div>
    )
  }
  return (
    <div className="auth-bar">
      <span className="auth-who muted">Play daily league puzzles with friends</span>
      <button
        type="button"
        className="btn btn-primary btn-sm"
        onClick={onSignIn}
        disabled={auth.status === 'loading'}
      >
        Sign in
      </button>
    </div>
  )
}

const THEME_PREVIEW_CARD = { count: 1, colour: 2, shape: 1, fill: 1 } as const

function ThemePicker({
  value,
  onChange,
}: {
  value: CardTheme
  onChange: (theme: CardTheme) => void
}) {
  return (
    <section className="theme-picker" aria-labelledby="theme-heading">
      <div className="theme-picker-head">
        <h2 id="theme-heading">Card style</h2>
        <span>{CARD_THEMES.find((theme) => theme.id === value)?.description}</span>
      </div>
      <div className="theme-options" role="group" aria-label="Card style">
        {CARD_THEMES.map((theme) => {
          const selected = theme.id === value
          return (
            <button
              type="button"
              className={`theme-option${selected ? ' is-active' : ''}`}
              aria-pressed={selected}
              onClick={() => onChange(theme.id)}
              key={theme.id}
            >
              <Card card={THEME_PREVIEW_CARD} theme={theme.id} label={`${theme.label} preview`} />
              <span>{theme.label}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}

function ModeCard({
  mode,
  title,
  blurb,
  played,
  meanMs,
  completion,
  onStart,
}: {
  mode: Mode
  title: string
  blurb: string
  played: number
  meanMs: number | null
  completion: number
  onStart: (mode: Mode) => void
}) {
  return (
    <button type="button" className="mode-card" onClick={() => onStart(mode)}>
      <span className="mode-card-title">{title}</span>
      <span className="mode-card-blurb">{blurb}</span>
      <span className="mode-card-stats">
        {played === 0
          ? 'No games yet'
          : `${played} completed · avg ${meanMs === null ? '—' : formatTime(meanMs)} · ${Math.round(
              completion * 100,
            )}% finished`}
      </span>
    </button>
  )
}

export function Menu({
  onStart,
  refreshKey,
  cardTheme,
  onCardThemeChange,
  auth,
  onSignIn,
  onSignOut,
  onSelectLeague,
}: MenuProps) {
  const records = useMemo(() => loadRecords(), [refreshKey])
  const aggA = aggregateSolveTimes(records, 'practice', 'A')
  const aggB = aggregateSolveTimes(records, 'practice', 'B')
  const aggC = aggregateSolveTimes(records, 'practice', 'C')

  return (
    <div className="menu">
      <header className="menu-head">
        <h1>Set</h1>
        <p className="muted">Practice — find every set on the board.</p>
      </header>

      <AuthBar auth={auth} onSignIn={onSignIn} onSignOut={onSignOut} />

      {auth.status === 'signedIn' && <LeaguesPanel onSelectLeague={onSelectLeague} />}

      <ThemePicker value={cardTheme} onChange={onCardThemeChange} />

      <div className="mode-cards">
        <ModeCard
          mode="A"
          title="Mode A · Six sets"
          blurb="The board has exactly 6 sets. You can see how many remain."
          played={aggA.gamesCompleted}
          meanMs={aggA.meanTotalTimeMs}
          completion={completionRate(records, 'practice', 'A')}
          onStart={onStart}
        />
        <ModeCard
          mode="B"
          title="Mode B · Unknown"
          blurb="An ordinary deal. You are not told how many sets exist."
          played={aggB.gamesCompleted}
          meanMs={aggB.meanTotalTimeMs}
          completion={completionRate(records, 'practice', 'B')}
          onStart={onStart}
        />
        <ModeCard
          mode="C"
          title="Mode C · Hardcore"
          blurb="Unknown count, no auto-finish. Hit Done when you think you’re done — a wrong Done adds a growing time penalty."
          played={aggC.gamesCompleted}
          meanMs={aggC.meanTotalTimeMs}
          completion={completionRate(records, 'practice', 'C')}
          onStart={onStart}
        />
      </div>
    </div>
  )
}
