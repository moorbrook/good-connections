import { useState, useCallback, type ReactNode } from 'react'
import { generateBoard, type Board, type Mode } from './game/board'
import { Menu } from './ui/Menu'
import { Game } from './ui/Game'
import { LeagueResult } from './ui/LeagueResult'
import { CardThemeProvider } from './ui/CardThemeContext'
import { loadCardTheme, saveCardTheme, type CardTheme } from './ui/cardThemes'
import { useAuth } from './ui/useAuth'
import { SignIn } from './ui/SignIn'
import { DisplayNamePrompt } from './ui/DisplayNamePrompt'
import { supabase } from './lib/supabase'
import { getTodayPuzzle, hasPlayedToday, type League, type TodayPuzzle } from './lib/leagues'

/**
 * App shell. Practice works signed-out; signing in (magic link) unlocks league
 * play — join by code, then today's shared puzzle and its leaderboard.
 */

interface Playing {
  board: Board
  gameId: number // remounts Game on "play again"
}

type LeagueView =
  | { phase: 'loading'; league: League }
  | { phase: 'error'; league: League; message: string }
  | { phase: 'play'; league: League; puzzle: TodayPuzzle; userId: string }
  | {
      phase: 'result'
      league: League
      puzzleDate: string
      userId: string
      /** False = arrived via "Standings" without playing; today stays hidden. */
      playedToday: boolean
    }

export function App() {
  const auth = useAuth()
  const [playing, setPlaying] = useState<Playing | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [cardTheme, setCardTheme] = useState<CardTheme>(loadCardTheme)
  const [showSignIn, setShowSignIn] = useState(false)
  const [league, setLeague] = useState<LeagueView | null>(null)

  const start = useCallback((mode: Mode) => {
    setPlaying({ board: generateBoard(mode, Math.random), gameId: Date.now() })
  }, [])

  const playAgain = useCallback(() => {
    setPlaying((prev) =>
      prev ? { board: generateBoard(prev.board.mode, Math.random), gameId: Date.now() } : prev,
    )
    setRefreshKey((k) => k + 1)
  }, [])

  const toMenu = useCallback(() => {
    setPlaying(null)
    setRefreshKey((k) => k + 1)
  }, [])

  const changeCardTheme = useCallback((theme: CardTheme) => {
    setCardTheme(theme)
    saveCardTheme(theme)
  }, [])

  /**
   * Open a league. `intent: 'standings'` goes to the results page WITHOUT
   * playing, so history is reachable on a day you haven't played — which it
   * previously was not, since the only route to the results page ran through
   * having already played. LeagueResult keeps today out of the date picker in
   * that case, so this is a read-only look at finished days.
   */
  const selectLeague = useCallback(async (l: League, intent: 'play' | 'standings' = 'play') => {
    setLeague({ phase: 'loading', league: l })
    try {
      const puzzle = await getTodayPuzzle(l.id)
      const played = await hasPlayedToday(l.id, puzzle.puzzleDate)
      const { data } = await supabase.auth.getUser()
      const userId = data.user?.id ?? ''
      if (played || intent === 'standings') {
        setLeague({
          phase: 'result',
          league: l,
          puzzleDate: puzzle.puzzleDate,
          userId,
          playedToday: played,
        })
      } else {
        setLeague({ phase: 'play', league: l, puzzle, userId })
      }
    } catch (e) {
      setLeague({ phase: 'error', league: l, message: e instanceof Error ? e.message : String(e) })
    }
  }, [])

  const exitLeague = useCallback(() => {
    setLeague(null)
    setRefreshKey((k) => k + 1)
  }, [])

  const wrap = (node: ReactNode) => <CardThemeProvider theme={cardTheme}>{node}</CardThemeProvider>

  if (auth.state.status === 'needsProfile') {
    return wrap(
      <DisplayNamePrompt email={auth.state.email} onSave={auth.saveDisplayName} onSignOut={auth.signOut} />,
    )
  }

  if (showSignIn && auth.state.status !== 'signedIn') {
    return wrap(<SignIn onSend={auth.sendMagicLink} onCancel={() => setShowSignIn(false)} />)
  }

  if (league) {
    if (league.phase === 'loading') {
      return wrap(
        <div className="auth-screen">
          <div className="auth-card">
            <h1>Loading…</h1>
            <p className="muted">Fetching today’s puzzle for {league.league.name}.</p>
          </div>
        </div>,
      )
    }
    if (league.phase === 'error') {
      return wrap(
        <div className="auth-screen">
          <div className="auth-card">
            <h1>Couldn’t load</h1>
            <p className="auth-error">{league.message}</p>
            <button type="button" className="btn btn-ghost" onClick={exitLeague}>
              Back
            </button>
          </div>
        </div>,
      )
    }
    if (league.phase === 'result') {
      return wrap(
        <LeagueResult
          leagueId={league.league.id}
          leagueName={league.league.name}
          puzzleDate={league.puzzleDate}
          timezone={league.league.timezone}
          mode={league.league.mode}
          userId={league.userId}
          playedToday={league.playedToday}
          onExit={exitLeague}
        />,
      )
    }
    return wrap(
      <Game
        board={league.puzzle.board}
        session={{
          kind: 'league',
          leagueId: league.league.id,
          leagueName: league.league.name,
          puzzleDate: league.puzzle.puzzleDate,
          timezone: league.league.timezone,
          userId: league.userId,
          penaltyBaseMs: league.league.penalty_base_ms,
        }}
        onExit={exitLeague}
      />,
    )
  }

  if (playing) {
    return wrap(
      <Game
        key={playing.gameId}
        board={playing.board}
        session={{ kind: 'practice' }}
        onExit={toMenu}
        onPlayAgain={playAgain}
      />,
    )
  }

  return wrap(
    <Menu
      onStart={start}
      refreshKey={refreshKey}
      cardTheme={cardTheme}
      onCardThemeChange={changeCardTheme}
      auth={auth.state}
      onSignIn={() => setShowSignIn(true)}
      onSignOut={auth.signOut}
      onSelectLeague={selectLeague}
    />,
  )
}
