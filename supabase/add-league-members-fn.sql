-- league_members(): let a member list who else is in their league.
-- Run in the Supabase SQL editor, top to bottom. 2026-08-20.
--
-- NON-DESTRUCTIVE and re-runnable (create or replace). No data is touched.
--
-- Why this is needed. The memberships RLS policy is
--   "a user reads their own memberships"  using (user_id = auth.uid())
-- so a member can see that *they* are in a league but never who else is. Two
-- features need the roster, and neither can be built without it:
--
--   1. Stable per-player colours on the pace charts. Colour has to be assigned
--      from the league's full roster, not from whoever happens to have played
--      today -- otherwise a player's colour shifts on days someone is absent,
--      which is exactly the bug this fixes.
--   2. The "everyone has played, so the set cards are safe to show" gate on the
--      day's summary. That is a comparison against the roster size.
--
-- Deliberately NOT a loosened RLS policy on memberships. A policy would expose
-- joined_at and every league_id row to anyone who can craft a query; this
-- returns bare user ids for one league the caller already belongs to, and
-- nothing else. Members can already read each other's display names and game
-- records, so the roster leaks nothing new.
--
-- security definer + a pinned search_path, matching join_league/today_puzzle.

create or replace function league_members (p_league uuid)
  returns setof uuid
  language sql
  security definer
  set search_path = public
as $$
  select m.user_id
  from memberships m
  where m.league_id = p_league
    -- The caller must be in the league themselves. Without this any signed-in
    -- user could enumerate the roster of any league whose id they learned.
    and exists (
      select 1 from memberships self
      where self.league_id = p_league and self.user_id = auth.uid()
    )
  -- Join order, NOT user_id order. The client assigns chart colours by position
  -- in this list, so the order has to be append-only: a new member must take
  -- the next free colour without disturbing anyone else's. Sorting by user_id
  -- would reshuffle every member who happens to sort after the newcomer, which
  -- is the exact bug the stable-colour work is fixing. user_id breaks ties so
  -- the order is still deterministic.
  order by m.joined_at, m.user_id;
$$;

-- ---------------------------------------------------------------------------
-- Check it. As a signed-in member this returns that league's user ids; for a
-- league you are not in it returns zero rows rather than an error, which is the
-- behaviour the client expects (it degrades to "roster unknown").
-- ---------------------------------------------------------------------------
-- select * from league_members('<a league id you belong to>');
