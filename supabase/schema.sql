-- Set — league play schema (slice 3, first milestone)
--
-- Run this in the Supabase SQL editor once the project exists. It creates the
-- tables + row-level-security (RLS) rules so that:
--   * players can only read their own records and their leagues' leaderboards,
--   * the daily board's SOLUTION is never readable by any client (only the
--     server-side Edge Function can see it), per the anti-cheat design.
--
-- Board generation and leaderboard math are NOT in here on purpose — they reuse
-- the pure TypeScript modules (generateBoard, deriveStats) so there is one
-- source of truth. See supabase/functions/ (Edge Function) and the app client.
--
-- Order: all tables first (so policies can reference each other), then the
-- policies, then the join-by-code function.

-- ===========================================================================
-- Tables
-- ===========================================================================

-- profiles: one row per signed-in user, holds the display name shown on boards.
create table if not exists profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text not null check (char_length(display_name) between 1 and 40),
  created_at timestamptz not null default now()
);

-- leagues: the config you set up (name, timezone, game mode, join code).
create table if not exists leagues (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  timezone text not null,                       -- IANA tz, e.g. 'Asia/Singapore'
  mode text not null check (mode in ('A', 'B', 'C')),
  join_code text not null unique,               -- players use this to join
  penalty_base_ms int not null default 5000,    -- Mode C: base penalty (doubles each false done)
  created_by uuid references auth.users (id),
  created_at timestamptz not null default now()
);

-- memberships: who belongs to which league.
create table if not exists memberships (
  league_id uuid not null references leagues (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  joined_at timestamptz not null default now(),
  primary key (league_id, user_id)
);

-- daily_seeds: one random seed per league per local date. The seed is issued
-- (via today_puzzle() below) only for the CURRENT local date, so everyone gets
-- the same board while future puzzles cannot be pre-computed. The board itself
-- is generated client-side from this seed by the pure generateBoard().
create table if not exists daily_seeds (
  league_id uuid not null references leagues (id) on delete cascade,
  puzzle_date date not null,                     -- league-local date
  seed bigint not null,
  created_at timestamptz not null default now(),
  primary key (league_id, puzzle_date)
);

-- game_records: the append-only telemetry event log, posted to the server.
-- Same shape as slice 1's localStorage records; `context` still separates
-- league from practice. Derived stats are computed on read (deriveStats), never
-- stored, so they cannot drift from the log.
create table if not exists game_records (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  context text not null check (context in ('league', 'practice')),
  league_id uuid references leagues (id) on delete set null,
  puzzle_date date,                              -- league-local date, for league games
  mode text not null check (mode in ('A', 'B', 'C')),
  total_sets int not null,
  started_at timestamptz not null,
  submitted_at timestamptz not null default now(),
  events jsonb not null
);

-- ===========================================================================
-- Row-level security
-- ===========================================================================

alter table profiles enable row level security;
alter table leagues enable row level security;
alter table memberships enable row level security;
alter table daily_seeds enable row level security;    -- no SELECT policy; reached via today_puzzle()
alter table game_records enable row level security;

-- profiles ------------------------------------------------------------------
create policy "profiles readable by any signed-in user"
  on profiles for select to authenticated using (true);
create policy "a user creates their own profile"
  on profiles for insert to authenticated with check (auth.uid() = id);
create policy "a user updates their own profile"
  on profiles for update to authenticated using (auth.uid() = id);

-- leagues -------------------------------------------------------------------
create policy "members can read their leagues"
  on leagues for select to authenticated using (
    exists (
      select 1 from memberships m
      where m.league_id = leagues.id and m.user_id = auth.uid()
    )
  );

-- memberships ---------------------------------------------------------------
create policy "a user reads their own memberships"
  on memberships for select to authenticated using (user_id = auth.uid());

-- game_records --------------------------------------------------------------
create policy "a user inserts their own records"
  on game_records for insert to authenticated with check (user_id = auth.uid());
create policy "a user reads their own records"
  on game_records for select to authenticated using (user_id = auth.uid());
create policy "members read their league's records"
  on game_records for select to authenticated using (
    context = 'league' and exists (
      select 1 from memberships m
      where m.league_id = game_records.league_id and m.user_id = auth.uid()
    )
  );

-- ===========================================================================
-- Join-by-code: enroll the caller without exposing the whole leagues table.
-- ===========================================================================
create or replace function join_league (code text)
  returns uuid
  language plpgsql
  security definer
  set search_path = public
as $$
declare
  target uuid;
begin
  select id into target from leagues where join_code = code;
  if target is null then
    raise exception 'no league with that code';
  end if;
  insert into memberships (league_id, user_id)
    values (target, auth.uid())
    on conflict do nothing;
  return target;
end;
$$;

-- ===========================================================================
-- today_puzzle: issue the current local-date seed for a league (members only),
-- creating it on first request. Returns { seed, puzzle_date, mode }. Only ever
-- issues today's seed, so future puzzles cannot be pre-computed.
-- ===========================================================================
create or replace function today_puzzle (p_league uuid)
  returns json
  language plpgsql
  security definer
  set search_path = public
as $$
declare
  tz text;
  m text;
  d date;
  s bigint;
begin
  select timezone, mode into tz, m from leagues where id = p_league;
  if tz is null then
    raise exception 'no such league';
  end if;
  if not exists (
    select 1 from memberships where league_id = p_league and user_id = auth.uid()
  ) then
    raise exception 'not a member of this league';
  end if;
  d := (now() at time zone tz)::date;
  insert into daily_seeds (league_id, puzzle_date, seed)
    values (p_league, d, (random() * 2000000000)::bigint)
    on conflict (league_id, puzzle_date) do nothing;
  select seed into s from daily_seeds where league_id = p_league and puzzle_date = d;
  return json_build_object('seed', s, 'puzzle_date', d, 'mode', m);
end;
$$;

-- ===========================================================================
-- league_members: the roster of one league, for members of that league only.
--
-- The memberships policy above is deliberately "your own rows only", so a
-- member cannot otherwise see who else is in their league. Two features need
-- the roster: stable per-player chart colours (assigned from the full roster,
-- so a colour does not shift on days someone is absent) and the "everyone has
-- played" gate that decides when a day's set cards become safe to show.
--
-- Returns bare user ids for a league the caller already belongs to, and
-- nothing else -- members can already read each other's display names and
-- game records, so this leaks nothing new. Zero rows for a league you are not
-- in, rather than an error.
-- ===========================================================================
create or replace function league_members (p_league uuid)
  returns setof uuid
  language sql
  security definer
  set search_path = public
as $$
  select m.user_id
  from memberships m
  where m.league_id = p_league
    and exists (
      select 1 from memberships self
      where self.league_id = p_league and self.user_id = auth.uid()
    )
  -- Join order, NOT user_id order: the client assigns chart colours by position
  -- here, so a new member must append rather than reshuffle everyone after them.
  order by m.joined_at, m.user_id;
$$;

-- ===========================================================================
-- past_puzzle: the seed for a league day that is already OVER, so the history
-- browser can redraw a finished board.
--
-- today_puzzle() only ever issues TODAY's seed and daily_seeds has no SELECT
-- policy, which keeps future puzzles unguessable but also puts finished boards
-- out of reach. This restores the finished ones and nothing else: p_date >=
-- today is refused, so today's board still comes only from today_puzzle() and
-- tomorrow's stays unreachable. Members only.
-- ===========================================================================
create or replace function past_puzzle (p_league uuid, p_date date)
  returns json
  language plpgsql
  security definer
  set search_path = public
as $$
declare
  tz text;
  m text;
  s bigint;
  today date;
begin
  select timezone, mode into tz, m from leagues where id = p_league;
  if tz is null then
    raise exception 'no such league';
  end if;
  if not exists (
    select 1 from memberships where league_id = p_league and user_id = auth.uid()
  ) then
    raise exception 'not a member of this league';
  end if;
  today := (now() at time zone tz)::date;
  if p_date >= today then
    raise exception 'that puzzle is not finished yet';
  end if;
  select seed into s from daily_seeds where league_id = p_league and puzzle_date = p_date;
  if s is null then
    return null;
  end if;
  return json_build_object('seed', s, 'puzzle_date', p_date, 'mode', m);
end;
$$;
