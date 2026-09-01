-- Add the CORSICA league; rename ORC-MODE's display name to "Orc Style".
-- Run in the Supabase SQL editor, top to bottom. 2026-08-19.
--
-- NON-DESTRUCTIVE and re-runnable. No game records, memberships or existing
-- league rows are deleted; running it twice leaves exactly the same state.
-- Existing players keep their memberships and do not need to re-join.
-- Players who want CORSICA join it with the code below, as normal.

-- ---------------------------------------------------------------------------
-- 1. CORSICA — Mode A, new board at 6pm US Eastern Standard Time.
--
-- A league's day rolls over at LOCAL MIDNIGHT in its timezone; the timezone
-- column is the only rollover control there is. So the zone to pick is the one
-- whose local midnight lands on the mark we want:
--
--   6pm EST = 23:00 UTC, and local midnight is 23:00 UTC in UTC+1 = Etc/GMT-1.
--
-- (The POSIX Etc/ zones invert the sign, so Etc/GMT-1 is UTC+1, not UTC-1.)
--
-- Etc/GMT-1 is a fixed offset that never observes DST, so the mark is a stable
-- instant all year — which is what makes solve times comparable across a
-- season. The consequence, stated plainly: US Eastern *does* observe DST, so
-- from mid-March to early November that same instant reads as 7pm EDT on an
-- Eastern wall clock. Pinning 6pm to the Eastern wall clock year-round instead
-- would need a rollover-hour column in `leagues`, not just a timezone.
-- There is a test in src/lib/time.test.ts pinning both readings.
-- ---------------------------------------------------------------------------
insert into leagues (name, timezone, mode, join_code, penalty_base_ms) values
  ('Corsica', 'Etc/GMT-1', 'A', 'CORSICA', 5000)
on conflict (join_code) do nothing;

-- ---------------------------------------------------------------------------
-- 2. Rename ORC-MODE for presentation. The join code is unchanged, so nobody
-- has to re-join and no existing membership or game record is affected — the
-- name is only ever read for display.
-- ---------------------------------------------------------------------------
update leagues set name = 'Orc Style' where join_code = 'ORC-MODE';

-- ---------------------------------------------------------------------------
-- 3. Check what you just made. Expect three rows: CORSICA, OG-SET, ORC-MODE.
-- ---------------------------------------------------------------------------
select
  name,
  join_code,
  mode,
  timezone,
  penalty_base_ms,
  -- The rollover mark, read on two clocks. These are the columns to eyeball:
  -- CORSICA should show 18:00 in New York in winter and 19:00 in summer.
  to_char(
    (date_trunc('day', (now() at time zone timezone)) + interval '1 day')
      at time zone timezone at time zone 'Asia/Singapore',
    'YYYY-MM-DD HH24:MI'
  ) as next_puzzle_sgt,
  to_char(
    (date_trunc('day', (now() at time zone timezone)) + interval '1 day')
      at time zone timezone at time zone 'America/New_York',
    'YYYY-MM-DD HH24:MI'
  ) as next_puzzle_nyc
from leagues
order by join_code;
