import { supabase } from './supabase'
import { generateBoard, mulberry32, type Board, type Mode } from '../game/board'
import { timeSpread, type TimeSpread } from '../game/pace'
import { zonedDateISO } from './time'
import {
  deriveStats,
  summarisePlayer,
  type GameRecord,
  type GameStats,
  type PlayerSummary,
  type TelemetryEvent,
} from '../game/telemetry'

/**
 * League data layer over Supabase. Board generation and stats stay in the pure
 * modules: today_puzzle() hands back a per-day seed and generateBoard() turns it
 * into the same board for everyone; the leaderboard runs deriveStats() over each
 * player's event log. The server stores no derived numbers, so nothing drifts.
 */

export interface League {
  id: string
  name: string
  timezone: string
  mode: Mode
  join_code: string
  /** Mode C base penalty in ms (nth false done costs base * 2^(n-1)). */
  penalty_base_ms: number
}

/** Leagues the signed-in user belongs to. */
export async function getMyLeagues(): Promise<League[]> {
  const { data, error } = await supabase
    .from('memberships')
    .select('leagues(id, name, timezone, mode, join_code, penalty_base_ms)')
  if (error) throw new Error(error.message)
  const rows = (data ?? []) as unknown as { leagues: League | League[] | null }[]
  const leagues: League[] = []
  for (const row of rows) {
    if (!row.leagues) continue
    if (Array.isArray(row.leagues)) leagues.push(...row.leagues)
    else leagues.push(row.leagues)
  }
  return leagues
}

/**
 * The league's members, in join order — the order chart colours are assigned
 * from, so it must stay append-only (see ui/playerColour).
 *
 * Returns null rather than throwing if the `league_members()` function is not
 * installed on the database yet. Every caller treats null as "roster unknown"
 * and degrades: colours fall back to a stable sort, and the set-card reveal
 * falls back to waiting for the day to close. A missing migration should soften
 * two features, not break the results page.
 */
export async function getLeagueRoster(leagueId: string): Promise<string[] | null> {
  const { data, error } = await supabase.rpc('league_members', { p_league: leagueId })
  if (error) return null
  const ids = (data ?? []) as unknown
  if (!Array.isArray(ids)) return null
  // The RPC returns a setof uuid, which supabase-js hands back as bare strings.
  return ids.map((v) => String(v))
}

/** Join a league by its code; returns the league id. */
export async function joinLeague(code: string): Promise<string> {
  const { data, error } = await supabase.rpc('join_league', { code: code.trim() })
  if (error) throw new Error(error.message)
  return data as string
}

/**
 * Every date this league has games on record for, newest first.
 *
 * RLS already lets a member read their league's records, so this needs no new
 * policy. It returns the raw list INCLUDING today — deciding what a given
 * member may actually look at is {@link viewableDates}'s job, kept separate so
 * the "today is not safe until you've played it" rule has one home.
 */
export async function getLeagueDates(leagueId: string): Promise<string[]> {
  const { data, error } = await supabase
    .from('game_records')
    .select('puzzle_date')
    .eq('context', 'league')
    .eq('league_id', leagueId)
  if (error) throw new Error(error.message)
  const dates = ((data ?? []) as { puzzle_date: string }[])
    .map((r) => r.puzzle_date)
    .filter((d): d is string => typeof d === 'string')
  return [...new Set(dates)].sort((a, b) => b.localeCompare(a))
}

export interface TodayPuzzle {
  seed: number
  puzzleDate: string // YYYY-MM-DD, the league's local date
  mode: Mode
  board: Board
}

/** Fetch today's seed for a league and build its board with the pure generator. */
export async function getTodayPuzzle(leagueId: string): Promise<TodayPuzzle> {
  const { data, error } = await supabase.rpc('today_puzzle', { p_league: leagueId })
  if (error) throw new Error(error.message)
  const row = data as { seed: number; puzzle_date: string; mode: Mode }
  const seed = Number(row.seed)
  const board = generateBoard(row.mode, mulberry32(seed))
  return { seed, puzzleDate: row.puzzle_date, mode: row.mode, board }
}

/**
 * Rebuild the board for a league day that is already over.
 *
 * Returns null when the day cannot be redrawn — no `past_puzzle()` function
 * installed, nobody ever played that day so no seed was issued, or the server
 * refused because the date is not actually finished. Every caller treats null
 * as "no cards for this day" and still shows the rest of the summary, so this
 * never has to be the difference between a page and an error.
 */
export async function getPastPuzzleBoard(leagueId: string, puzzleDate: string): Promise<Board | null> {
  const { data, error } = await supabase.rpc('past_puzzle', {
    p_league: leagueId,
    p_date: puzzleDate,
  })
  if (error || data === null || data === undefined) return null
  const row = data as { seed: number; mode: Mode }
  return generateBoard(row.mode, mulberry32(Number(row.seed)))
}

/** Post a completed/abandoned league game's event log to the server. */
export async function submitLeagueRecord(args: {
  leagueId: string
  puzzleDate: string
  mode: Mode
  totalSets: number
  startedAtMs: number
  events: TelemetryEvent[]
}): Promise<void> {
  const { data: userData } = await supabase.auth.getUser()
  const userId = userData.user?.id
  if (!userId) throw new Error('not signed in')
  const { error } = await supabase.from('game_records').insert({
    user_id: userId,
    context: 'league',
    league_id: args.leagueId,
    puzzle_date: args.puzzleDate,
    mode: args.mode,
    total_sets: args.totalSets,
    started_at: new Date(args.startedAtMs).toISOString(),
    events: args.events,
  })
  if (error) throw new Error(error.message)
}

export interface LeaderboardRow {
  userId: string
  displayName: string
  stats: GameStats
  /** Raw event log, so a per-player solve timeline can be rendered. */
  events: TelemetryEvent[]
}

/** Today's standings for a league: every member's game, ranked by solve time. */
export async function getLeaderboard(leagueId: string, puzzleDate: string): Promise<LeaderboardRow[]> {
  const { data: recs, error } = await supabase
    .from('game_records')
    .select('user_id, events')
    .eq('context', 'league')
    .eq('league_id', leagueId)
    .eq('puzzle_date', puzzleDate)
  if (error) throw new Error(error.message)
  const records = (recs ?? []) as { user_id: string; events: TelemetryEvent[] }[]

  const userIds = [...new Set(records.map((r) => r.user_id))]
  const names = new Map<string, string>()
  if (userIds.length > 0) {
    const { data: profs } = await supabase.from('profiles').select('id, display_name').in('id', userIds)
    for (const p of (profs ?? []) as { id: string; display_name: string }[]) {
      names.set(p.id, p.display_name)
    }
  }

  const rows: LeaderboardRow[] = records.map((r) => ({
    userId: r.user_id,
    displayName: names.get(r.user_id) ?? '—',
    stats: deriveStats({ events: r.events } as GameRecord),
    events: r.events,
  }))

  // Completed games first, fastest total time first; unfinished games after.
  rows.sort((a, b) => {
    if (a.stats.completed !== b.stats.completed) return a.stats.completed ? -1 : 1
    return (a.stats.totalTimeMs ?? Infinity) - (b.stats.totalTimeMs ?? Infinity)
  })
  // One row per player — their best attempt (first after the sort).
  const seen = new Set<string>()
  return rows.filter((r) => (seen.has(r.userId) ? false : (seen.add(r.userId), true)))
}

// --- league-wide records / member stats (all-time) -------------------------

export interface SoloRecord {
  displayName: string
  timeMs: number
  puzzleDate: string
}

export interface MemberStat {
  userId: string
  displayName: string
  gamesPlayed: number
  gamesCompleted: number
  completionRate: number
  bestTimeMs: number | null
  /** Best/worst/mean/std-dev over COMPLETED games only — abandoned games have
   * a shorter elapsed time, so including them would reward giving up. Null
   * until the member has finished at least one. */
  spread: TimeSpread | null
  /** Total penalty time this member has been charged across the league. */
  penaltyMs: number
  /** Consecutive calendar days completed, ending at the member's latest. */
  currentStreak: number
}

/** A "hall of fame/shame" superlative — the single game that maxed some metric. */
export interface NotableRecord {
  label: string
  displayName: string
  value: number
  unit: 'count' | 'time'
  puzzleDate: string
}

export interface LeagueStats {
  mode: Mode
  /** Mode A: top-3 fastest completed solves. */
  topSolves: SoloRecord[]
  /** Mode B: fastest completed solve for each board set-count. */
  fastestBySetCount: { setCount: number; record: SoloRecord }[]
  members: MemberStat[]
  notables: NotableRecord[]
}

function prevDateISO(iso: string): string {
  const [y, m, d] = iso.split('-').map(Number)
  const dt = new Date(Date.UTC(y, m - 1, d))
  dt.setUTCDate(dt.getUTCDate() - 1)
  return dt.toISOString().slice(0, 10)
}

function currentStreak(completedDates: string[]): number {
  if (completedDates.length === 0) return 0
  const days = new Set(completedDates)
  const latest = [...days].sort()[days.size - 1]!
  let streak = 0
  let day = latest
  while (days.has(day)) {
    streak++
    day = prevDateISO(day)
  }
  return streak
}

/** All-time records + per-member stats for a league. */
export async function getLeagueStats(leagueId: string, mode: Mode): Promise<LeagueStats> {
  const { data, error } = await supabase
    .from('game_records')
    .select('user_id, puzzle_date, total_sets, events')
    .eq('context', 'league')
    .eq('league_id', leagueId)
  if (error) throw new Error(error.message)
  const recs = (data ?? []) as {
    user_id: string
    puzzle_date: string
    total_sets: number
    events: TelemetryEvent[]
  }[]

  const userIds = [...new Set(recs.map((r) => r.user_id))]
  const names = new Map<string, string>()
  if (userIds.length > 0) {
    const { data: profs } = await supabase.from('profiles').select('id, display_name').in('id', userIds)
    for (const p of (profs ?? []) as { id: string; display_name: string }[]) {
      names.set(p.id, p.display_name)
    }
  }

  const games = recs.map((r) => ({
    userId: r.user_id,
    name: names.get(r.user_id) ?? '—',
    date: r.puzzle_date,
    setCount: r.total_sets,
    stats: deriveStats({ events: r.events } as GameRecord),
  }))
  const completed = games.filter((g) => g.stats.completed && g.stats.totalTimeMs !== null)

  const topSolves: SoloRecord[] = [...completed]
    .sort((a, b) => a.stats.totalTimeMs! - b.stats.totalTimeMs!)
    .slice(0, 3)
    .map((g) => ({ displayName: g.name, timeMs: g.stats.totalTimeMs!, puzzleDate: g.date }))

  const byCount = new Map<number, SoloRecord>()
  for (const g of completed) {
    const cur = byCount.get(g.setCount)
    if (!cur || g.stats.totalTimeMs! < cur.timeMs) {
      byCount.set(g.setCount, { displayName: g.name, timeMs: g.stats.totalTimeMs!, puzzleDate: g.date })
    }
  }
  const fastestBySetCount = [...byCount.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([setCount, record]) => ({ setCount, record }))

  const byUser = new Map<string, typeof games>()
  for (const g of games) {
    const arr = byUser.get(g.userId) ?? []
    arr.push(g)
    byUser.set(g.userId, arr)
  }
  const members: MemberStat[] = [...byUser.entries()]
    .map(([userId, gs]) => {
      const comp = gs.filter((g) => g.stats.completed && g.stats.totalTimeMs !== null)
      const spread = timeSpread(comp.map((g) => g.stats.totalTimeMs!))
      return {
        userId,
        displayName: names.get(userId) ?? '—',
        gamesPlayed: gs.length,
        gamesCompleted: comp.length,
        completionRate: gs.length > 0 ? comp.length / gs.length : 0,
        bestTimeMs: spread?.bestMs ?? null,
        spread,
        // Every game, not just completed ones: a penalty was really paid even
        // in a game the player went on to abandon.
        penaltyMs: gs.reduce((acc, g) => acc + g.stats.penaltyMs, 0),
        currentStreak: currentStreak(comp.map((g) => g.date)),
      }
    })
    .sort((a, b) => (a.bestTimeMs ?? Infinity) - (b.bestTimeMs ?? Infinity))

  const notable = (
    label: string,
    unit: 'count' | 'time',
    metric: (g: (typeof games)[number]) => number,
  ): NotableRecord | null => {
    let best: { g: (typeof games)[number]; value: number } | null = null
    for (const g of games) {
      const v = metric(g)
      if (v > 0 && (best === null || v > best.value)) best = { g, value: v }
    }
    return best
      ? { label, displayName: best.g.name, value: best.value, unit, puzzleDate: best.g.date }
      : null
  }
  const notables = [
    notable('Most wrong guesses in a game', 'count', (g) => g.stats.errorCount),
    notable('Most repeat picks in a game', 'count', (g) => g.stats.duplicateCount),
    notable('Longest stall between sets', 'time', (g) =>
      g.stats.setIntervalsMs.length > 0 ? Math.max(...g.stats.setIntervalsMs) : 0,
    ),
    notable('Most premature “done”s', 'count', (g) => g.stats.falseDones),
    notable('Most time lost to penalties', 'time', (g) => g.stats.penaltyMs),
  ].filter((n): n is NotableRecord => n !== null)

  return { mode, topSolves, fastestBySetCount, members, notables }
}

// --- one player's history (the performance page) ---------------------------

export interface PlayerGame {
  /** game_records row id; ties a summary superlative back to its game. */
  id: string
  puzzleDate: string
  /** Sets on that day's board. */
  totalSets: number
  stats: GameStats
  events: TelemetryEvent[]
}

export interface PlayerHistory {
  userId: string
  displayName: string
  /** Newest puzzle date first. */
  games: PlayerGame[]
  summary: PlayerSummary
}

/**
 * Every league game one member has played, plus the summary derived from them.
 * Any member may look at any other member — RLS already allows reading the
 * league's records, and nothing here is a secret once a day has been played.
 */
export async function getPlayerHistory(
  leagueId: string,
  userId: string,
  mode: Mode,
): Promise<PlayerHistory> {
  const { data, error } = await supabase
    .from('game_records')
    .select('id, puzzle_date, total_sets, mode, started_at, events')
    .eq('context', 'league')
    .eq('league_id', leagueId)
    .eq('user_id', userId)
  if (error) throw new Error(error.message)
  const rows = (data ?? []) as {
    id: string
    puzzle_date: string
    total_sets: number
    mode: Mode
    started_at: string
    events: TelemetryEvent[]
  }[]

  const { data: prof } = await supabase
    .from('profiles')
    .select('display_name')
    .eq('id', userId)
    .maybeSingle()

  // Newest first, once — the page reads in that order and so do the
  // superlatives, whose ties then fall to the more recent game whatever order
  // the rows arrived in.
  rows.sort((a, b) => b.puzzle_date.localeCompare(a.puzzle_date))

  const records: GameRecord[] = rows.map((r) => ({
    id: r.id,
    player: userId,
    context: 'league',
    mode: r.mode,
    totalSets: r.total_sets,
    startedAtMs: Date.parse(r.started_at),
    events: r.events,
  }))

  const games: PlayerGame[] = rows.map((r, i) => ({
    id: r.id,
    puzzleDate: r.puzzle_date,
    totalSets: r.total_sets,
    stats: deriveStats(records[i]!),
    events: r.events,
  }))

  return {
    userId,
    displayName: (prof as { display_name: string } | null)?.display_name ?? '—',
    games,
    summary: summarisePlayer(records, 'league', mode),
  }
}

/**
 * For each league, whether the signed-in user has already played its CURRENT
 * puzzle — so the menu can show at a glance which leagues have a fresh game
 * waiting.
 *
 * The league-local date is computed client-side with the same rule the server
 * uses in today_puzzle() (`(now() at time zone tz)::date`), which is what makes
 * a league with a non-midnight rollover come out right: CORSICA's day turns at
 * 6pm US Eastern, not at the viewer's midnight.
 *
 * One query for every league rather than one each. A league whose timezone the
 * browser cannot parse is reported as unplayed — "there might be a game" is the
 * harmless direction to be wrong in.
 */
export async function getPlayedToday(
  leagues: readonly { id: string; timezone: string }[],
): Promise<Record<string, boolean>> {
  const out: Record<string, boolean> = {}
  const dates = new Map<string, string>()
  for (const l of leagues) {
    const d = zonedDateISO(l.timezone, Date.now())
    out[l.id] = false
    if (d !== null) dates.set(l.id, d)
  }
  if (dates.size === 0) return out

  const { data: userData } = await supabase.auth.getUser()
  const userId = userData.user?.id
  if (!userId) return out

  const { data, error } = await supabase
    .from('game_records')
    .select('league_id, puzzle_date')
    .eq('context', 'league')
    .eq('user_id', userId)
    .in('league_id', [...dates.keys()])
    .in('puzzle_date', [...new Set(dates.values())])
  if (error) return out

  // Two leagues can share a date, so match on the PAIR — filtering by the two
  // `in` lists separately would mark a league played on another league's date.
  const played = new Set(
    ((data ?? []) as { league_id: string; puzzle_date: string }[]).map(
      (r) => `${r.league_id}|${r.puzzle_date}`,
    ),
  )
  for (const [leagueId, date] of dates) {
    out[leagueId] = played.has(`${leagueId}|${date}`)
  }
  return out
}

/** Has the signed-in user already submitted a game for this league + date? */
export async function hasPlayedToday(leagueId: string, puzzleDate: string): Promise<boolean> {
  const { data: userData } = await supabase.auth.getUser()
  const userId = userData.user?.id
  if (!userId) return false
  const { data, error } = await supabase
    .from('game_records')
    .select('id')
    .eq('context', 'league')
    .eq('league_id', leagueId)
    .eq('puzzle_date', puzzleDate)
    .eq('user_id', userId)
    .limit(1)
  if (error) throw new Error(error.message)
  return (data ?? []).length > 0
}
