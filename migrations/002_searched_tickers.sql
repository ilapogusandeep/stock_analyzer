-- StockIQ ticker autocomplete -- Supabase schema for searched/non-curated tickers.
--
-- Run once in your Supabase SQL Editor (after 001_predictions.sql).
--
-- Purpose: when a user analyzes a ticker that isn't in the SEC US equity
-- list (BTC-USD, ^GSPC, EURUSD=X, foreign ADRs, etc.), we remember it
-- here so it shows up in the dropdown next session, even after a
-- Streamlit Cloud redeploy.
--
-- Cost on the free tier: each row is ~150 bytes; even 10K unique
-- tickers = 1.5MB out of the 500MB DB allowance. One UPSERT per
-- analyze() call; one bulk SELECT on each dropdown render. Well within
-- limits for personal-app traffic.

CREATE TABLE IF NOT EXISTS searched_tickers (
    ticker        TEXT PRIMARY KEY,
    company_name  TEXT NOT NULL,
    first_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_searched_tickers_last_seen
    ON searched_tickers(last_seen DESC);

-- Personal single-user app -- disable RLS so the publishable/anon key
-- can read+write. If you later open the app to other users, re-enable
-- RLS and add policies.
ALTER TABLE searched_tickers DISABLE ROW LEVEL SECURITY;
