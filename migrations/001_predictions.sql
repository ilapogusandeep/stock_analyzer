-- StockIQ prediction track record — Supabase schema.
--
-- Run once in your Supabase project:
--   1. Open https://app.supabase.com -> your project -> SQL Editor
--   2. Paste this file's contents and click Run
--   3. Copy your project's URL + anon key from Project Settings > API
--   4. Add to Streamlit Cloud secrets as SUPABASE_URL and SUPABASE_KEY
--
-- After this lands, every analysis logs a prediction row, and old enough
-- predictions are scored automatically against actual price moves so the
-- Track Record panel shows real calibration data that survives redeploys.

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    -- prediction snapshot
    ticker                  TEXT             NOT NULL,
    timestamp               TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    direction               TEXT             NOT NULL,  -- BULLISH / NEUTRAL / BEARISH
    confidence              DOUBLE PRECISION NOT NULL,  -- 0..1
    bullish_prob            DOUBLE PRECISION,
    neutral_prob            DOUBLE PRECISION,
    bearish_prob            DOUBLE PRECISION,
    price_at_prediction     DOUBLE PRECISION NOT NULL,
    price_target            DOUBLE PRECISION,
    resolution_horizon_days INT              NOT NULL,
    -- resolution fields, populated lazily after horizon has passed
    resolved_at             TIMESTAMPTZ,
    price_at_resolution     DOUBLE PRECISION,
    actual_return           DOUBLE PRECISION,
    hit                     BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker     ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp  ON predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(timestamp) WHERE hit IS NULL;

-- Personal single-user app: disable RLS so the anon key can read/write.
-- If you want stricter security later, re-enable RLS and add policies.
ALTER TABLE predictions DISABLE ROW LEVEL SECURITY;
