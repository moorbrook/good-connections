-- past_puzzle(): hand back the seed for a league day that is already OVER.
-- Run in the Supabase SQL editor, top to bottom. 2026-08-20.
--
-- NON-DESTRUCTIVE and re-runnable (create or replace). No data is touched.
--
-- Why. today_puzzle() deliberately only ever issues TODAY's seed, and
-- daily_seeds has no SELECT policy at all, so a past board cannot be rebuilt by
-- any client. That is correct for keeping future puzzles unguessable, but it
-- also means the history browser can show a past day's leaderboard, letters and
-- pace chart while being unable to show the cards those letters refer to.
--
-- This closes that gap for finished days only. The guard is the whole point:
--
--   p_date >= today  ->  refused.
--
-- So today's board still comes only from today_puzzle(), which is what enforces
-- "you cannot see the board before you play it", and tomorrow's is still
-- unreachable. A day whose slot has closed is history — every member has had
-- their shot at it, and there is nothing left to leak.
--
-- Members of the league only, like every other function here.

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

  -- The league's own local date, exactly as today_puzzle() computes it, so a
  -- league with a non-midnight rollover is judged on its own clock.
  today := (now() at time zone tz)::date;
  if p_date >= today then
    raise exception 'that puzzle is not finished yet';
  end if;

  select seed into s from daily_seeds where league_id = p_league and puzzle_date = p_date;
  if s is null then
    return null;   -- a day nobody played: no seed was ever issued
  end if;
  return json_build_object('seed', s, 'puzzle_date', p_date, 'mode', m);
end;
$$;

-- ---------------------------------------------------------------------------
-- Check it. The first should return a seed; the second must raise
-- "that puzzle is not finished yet".
-- ---------------------------------------------------------------------------
-- select past_puzzle('<league id>', current_date - 1);
-- select past_puzzle('<league id>', current_date);
