-- StockIQ watchlist -- Supabase schema for the Scanner view's watchlist.
--
-- Run once in your Supabase SQL Editor (after 001_predictions.sql and
-- 002_searched_tickers.sql).
--
-- Purpose: persist the list of tickers a user is following so the
-- Scanner tab's watchlist signals (price delta, unusual options
-- count, aggressor balance, news velocity) carry across redeploys.
--
-- Cost on the free tier: each row is ~80 bytes; a watchlist of 100
-- names is 8KB. One INSERT per Add, one DELETE per Remove, one bulk
-- SELECT per Scanner render.

CREATE TABLE IF NOT EXISTS watchlist (
    ticker     TEXT PRIMARY KEY,
    added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Personal single-user app -- disable RLS so the publishable/anon key
-- can read/write. Re-enable RLS and add policies later if you open
-- the app to multiple users.
ALTER TABLE watchlist DISABLE ROW LEVEL SECURITY;
