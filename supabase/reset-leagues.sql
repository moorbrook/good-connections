-- Wipe every league and start again with three: OG-SET, ORC-MODE and CORSICA.
-- Run in the Supabase SQL editor, top to bottom. 2026-08-18, amended 2026-08-19.
--
-- This is the from-scratch script. It has already been run once (2026-08-18,
-- with the first two leagues only); the additive follow-up that added CORSICA
-- and renamed ORC-MODE on the live database is add-corsica-league.sql. Both
-- files must describe the same three leagues — if you change one, change the
-- other, or a future reset will silently undo the follow-up.
--
-- DESTRUCTIVE. This deletes every recorded league game, every league, and
-- every membership. It is deliberate: the board generator changed (Mode B/C
-- now reject sparse boards), so games recorded earlier sit on boards the
-- generator would no longer produce, and there is no generator versioning.
-- Practice records are untouched — they live in each device's localStorage,
-- never in this database.
--
-- AFTER RUNNING THIS, EVERY PLAYER MUST RE-JOIN with the new codes below.
-- Memberships are wiped along with the leagues, including yours.

-- ---------------------------------------------------------------------------
-- 1. Delete the league game records FIRST.
--
-- game_records.league_id is ON DELETE SET NULL, not CASCADE — so dropping the
-- leagues first would leave orphan rows with league_id = null that no query
-- ever reaches again. Delete them explicitly while they can still be found.
-- ---------------------------------------------------------------------------
delete from game_records where context = 'league';

-- ---------------------------------------------------------------------------
-- 2. Delete the leagues. This cascades to memberships and daily_seeds.
-- ---------------------------------------------------------------------------
delete from leagues;

-- ---------------------------------------------------------------------------
-- 3. Create the three new leagues.
--
-- A league's day rolls over at LOCAL MIDNIGHT in its timezone, so the timezone
-- column is what sets the "new game" mark.
--
--   OG-SET   -> Asia/Singapore : midnight SGT.
--   ORC-MODE -> Etc/GMT+7      : 3pm SGT.
--   CORSICA  -> Etc/GMT-1      : 6pm US Eastern Standard Time.
--
-- The POSIX Etc/ zones invert the sign, so Etc/GMT+7 is UTC-7 and Etc/GMT-1 is
-- UTC+1. Neither observes DST, so both marks are stable instants all year:
-- Etc/GMT+7's local midnight is 15:00 in Singapore, and Etc/GMT-1's is 23:00
-- UTC, which is 18:00 EST. A DST-observing zone at the same offset would drift
-- the mark by an hour twice a year — America/Phoenix happens to work only
-- because Arizona skips DST, and America/Los_Angeles would not.
--
-- The flip side of a fixed offset is that US Eastern itself is not fixed, so
-- CORSICA's mark reads as 7pm EDT from mid-March to early November. Holding
-- 6pm on the Eastern wall clock year-round would need a rollover-hour column,
-- not just a timezone. There are tests pinning all of these in Jan and July.
-- ---------------------------------------------------------------------------
insert into leagues (name, timezone, mode, join_code, penalty_base_ms) values
  ('OG Set',    'Asia/Singapore', 'A', 'OG-SET',   5000),
  ('Orc Style', 'Etc/GMT+7',      'C', 'ORC-MODE', 5000),
  ('Corsica',   'Etc/GMT-1',      'A', 'CORSICA',  5000);

-- ---------------------------------------------------------------------------
-- 4. Check what you just made.
-- ---------------------------------------------------------------------------
select
  name,
  join_code,
  mode,
  timezone,
  penalty_base_ms,
  -- The mark, read on two clocks — these are the columns to eyeball.
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
