# StockIQ

A single-page stock analysis dashboard. Pulls free public data (yfinance, RSS news, 13F filings, options chains), runs multi-horizon ML forecasts, and renders everything on one screen — no tabs, no page turns.

Live: **[stockiq.streamlit.app](https://stockiq.streamlit.app)** · Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) · Tests: [tests/README.md](tests/README.md)

---

## What's on the page

Every analysis renders four vertical columns over a single compact header band.

### Header band

- Ticker + company name + **one-liner**: `Industry · HQ City, State · N employees · Business summary`
- Live price + day change
- **Model signal** pill: BULLISH / NEUTRAL / BEARISH (1-week horizon)
- Market cap · sector · price target · **next earnings** date + countdown
- 7 external research links (Yahoo Finance, Finviz, TradingView, Stock Analysis, SEC Filings, Google News, Seeking Alpha, Website)

### Column 1 — Fundamentals

| Panel | What it shows |
|---|---|
| **Valuation** | P/E · Fwd P/E · PEG · P/B · P/S · Beta |
| **Health & Range** | Current Ratio · D/E · ROE · Margin · Rev Growth · 52w High/Low/From High/From Low |
| **Positioning** | Short % Float · Days to Cover · Short MoM Δ · Insider Own · Institutional Own |
| **Technical** | RSI (14) · MACD · SMA 20/50/200 · Volatility 20d |
| **Options Flow** | Horizontal pill strip: P/C Vol · P/C OI · Call/Put volume · ATM IV · Tilt |

### Column 2 — Chart + market view

- **Price/Volume/RSI chart** (1Y, shared x-axis) with SMA 20/50/200 + Bollinger bands overlay
- **Performance pills** (1D/5D/1M/3M/1Y returns) embedded in the chart header
- **Analyst & Sentiment** panel combining analyst consensus (rating, target mean, implied upside) with multi-source sentiment (overall + news score)

### Column 3 — Smart money

| Panel | What it shows |
|---|---|
| **Top Holders** | Latest 13F filings — top 8 institutions by value, with % held and quarterly position change, sorted by filing date |
| **Unusual Options** | Strikes where today's V/OI ≥ 2× across the next 3 expiries, ranked by dollar premium flow. Each row shows: direction · strike · expiry · V/OI ratio · premium · **aggressor tag** (B/S/M from bid-ask heuristic) · 🔗 cluster flag |
| **Earnings History** | Last 4 quarters — estimate, actual, surprise % |

### Column 4 — AI forecasts + news

- **AI · 1 week** — bullish/neutral/bearish probability bars + target prices + accuracy readout
- **AI · 1 month** — same structure, 21-day horizon
- **AI · 3 month regime** — 3-class classifier (BULLISH / SIDEWAYS / BEARISH) with probability mix, since a point price target on a single ticker at 3 months is basically noise
- **News feed** — up to 20 headlines sorted latest-first, with real publisher labels (Motley Fool, Benzinga, MarketBeat, Seeking Alpha, IBD, CNBC, Reuters, ...), recency-weighted sentiment scores, scrollable inline

---

## What makes this different from a Yahoo Finance page

### Honest walk-forward accuracy

Every forecast panel shows its test accuracy from a **chronological** 80/20 split — first 80% of history trains, last 20% tests. No random shuffling, no lookahead leakage. When you see `accuracy 53%` it's what the model *would have* scored on the most recent slice of real history.

### Live calibration feedback loop

Predictions land in a Supabase table (`predictions`) with `hit = NULL`. Once the resolution horizon (5d / 21d / 63d) passes, the next analysis call fetches the actual price, compares to the prediction, and updates `hit`. Once you have ≥ 5 resolved predictions per horizon, the UI **shrinks displayed confidence** toward the no-edge baseline by a factor scaled to the model's demonstrated edge:

- 75% hit rate → halfway shrinkage (the model gets roughly what it "earned")
- 50% hit rate → full shrinkage to uniform (UI says "we don't know")
- Below 50% → also shrinks (noise, not necessarily inverted signal)

Subtitle of each panel shows `calibrated · n=N` or `untracked · n=N` so you always know which mode it's in.

### Aggressor-inferred options direction (free-tier)

yfinance only gives end-of-day options snapshots — no real trade-by-trade bid/ask side. We infer aggressor by checking where `lastPrice` sits in the bid/ask spread:

- Top third of spread → **BUY** (buyer hit the ask)
- Bottom third → **SELL** (seller hit the bid)
- Middle → **MID** (inconclusive)

Not as sharp as Polygon's $29/mo time-and-sales, but significantly better than pure V/OI, and free.

### Real publisher names instead of "Google News" blob

We parse the `<source>` tag from every Google News RSS item and cross-fetch yfinance's native `ticker.news` (which returns publisher display names). Result: headlines labeled with the real outlet — Motley Fool, Benzinga, MarketBeat, Barron's (when cross-fed by another aggregator), IBD, Seeking Alpha, etc. — and source diversity becomes an ML feature (`news_source_count`).

---

## ML features feeding the forecasts

- **Price/technical**: returns (1d/5d/20d), RSI, MACD, Bollinger bands, SMA 5/20/50, volume ratio, 20d volatility
- **News**: overall sentiment, news count, positive ratio, confidence, **source diversity count**, **recency-weighted sentiment** (2-day half-life)
- **Social**: social sentiment score, mentions, engagement rate
- **Analyst**: rating mean, target mean, opinion count
- **Options**: P/C ratio, options sentiment label
- **Institutional**: ownership %, institution count, **13F net flow** (value-weighted pct_change across top holders)
- **Peer/sector relative strength**: rolling 1m and 3m relative return vs sector ETF (XLK/XLF/XLV/XLY/XLP/XLI/XLE/XLC/XLU/XLRE/XLB)
- **Earnings proximity**: days to next earnings (capped at 60), `earnings_imminent` (≤ 14 days)
- **Economic**: VIX, treasury yield, dollar index
- **Rankings**: Finviz ratios, MarketWatch ratings, Yahoo analyst stats, sector performance

~35 features in total feed an ensemble of RandomForest (60%) + GradientBoosting (40%).

---

---

## The Scanner view

A second top-level mode toggleable from the top bar (radio at right): switches the page from the per-ticker Analyze dashboard into a market-scanner that surfaces opportunities. Layout is **two side-by-side panels** (Watchlist on the left, Top movers on the right) plus a **bottom-width earnings strip**. Both tables are scrollable with sticky headers so the columns stay legible as you page through.

### Watchlist

User-curated list of tickers persisted to Supabase (`watchlist` table) so it survives redeploys. For each ticker the scanner pulls a **lightweight signal snapshot**:

| Signal | What it measures |
|---|---|
| Price + 1-day Δ | spot move |
| Unusual options count | strikes with V/OI ≥ 2× across the next 3 expiries |
| Aggressor net | direction-aware sum across unusual rows: BUY-CALL or SELL-PUT = **+1** (bullish), SELL-CALL or BUY-PUT = **−1** (bearish), MID = 0. Pairing aggressor with option type matters — heavy put-buying is bearish, not bullish, even though every row says "BUY" |
| News velocity | today's article count vs 7-day trailing avg per ticker |
| Composite score | 0–100 weighted blend, clamped per-axis at 25 so no single dimension dominates |
| Bias | BULLISH / NEUTRAL / BEARISH from aggressor + change_1d agreement |

**Adding a ticker**: type in the input below the table and hit `+ Add`. The Add path tries (in order) (1) the literal symbol against our merged universe, (2) yfinance's resolver for any symbol it recognizes (crypto `BTC-USD`, indices `^GSPC`, forex `EURUSD=X`), then (3) **fuzzy company-name match** against the merged universe — so typing `Oracle` resolves to `ORCL`, `Apple` to `AAPL`, `tesla` to `TSLA`. If none of those resolve, you get a friendly error explaining the symbol formats.

**Removing a ticker**: each row has an inline `×` pill in the rightmost column — single click removes it. Rows whose symbol returns no price data render dimmed with a red "no data" tag so you can spot typos / delistings at a glance.

Manual "🔄 Refresh" button beside the Add input forces a re-pull (results are cached 10 min by default).

### Top movers (curated universe)

Same lightweight scan run across ~50 hand-picked names — mega-cap tech + semis + finance + popular ETFs + your watchlist + recent searches (capped). Sorted by composite score; **top 30** surface in the table (was 15 — bumped to fill the side-by-side layout). Both watchlist and top-movers tables cap at ~520px height with internal scroll + sticky headers, so the page stays compact without truncating data.

Honest caveats:
- The aggressor signal is a free-tier heuristic (yfinance gives end-of-day snapshots, not trade prints) — same caveats as the Analyze view's Unusual Options panel.
- Scan results cached **10 minutes** to stay clear of yfinance rate limits. The first user pays the cost; subsequent renders are instant.
- The Top movers list is opinionated about the universe. To scan a wider net you'd want Polygon ($29/mo) and a real per-ticker rate-limit budget.

Click any ticker pill in the scanner → routes back to the Analyze view via `?ticker=` query param.

### Upcoming earnings (full-width band below the two tables)

A horizontal strip of the next 7 days of earnings prints across **watchlist ∪ ~50 curated names**. Each card shows ticker · day-of-week + M/D · time-to-event pill (`today` red / `tomorrow` or `in 2-3d` yellow / `in 4-7d` indigo) · estimated EPS when yfinance has it; clicking a card routes you to that ticker's Analyze dashboard.

Implementation details that matter:
- **Hidden when empty** — if no curated names report in the next 7 days the strip doesn't render, keeping the page clean.
- **Per-ticker `@st.cache_data` ttl=86400** — earnings dates don't change intraday, so day-2+ Scanner loads pay zero yfinance cost. The first cold load takes ~30s for ~50 tickers (you'll see a "Scanning upcoming earnings…" spinner).
- **Non-equity pre-filter** — ETFs (SPY/QQQ/DIA/XLK/...), indexes (`^VIX`), forex (`*=X`), and crypto (`*-USD`) short-circuit before any yfinance call. They don't report earnings and were spamming logs with "no earnings dates" warnings.
- **Reuses the dict/DataFrame fallback dance** from `stockiq.core.analyzer.get_earnings_calendar` — yfinance's `ticker.calendar` API has changed return types over time, so we handle both, then fall back to `ticker.earnings_dates` for the rest.

---

## Setup

### Run locally

```bash
git clone https://github.com/ilapogusandeep/stock_analyzer.git
cd stock_analyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run sidebar_web.py
```

No API keys required for basic operation — yfinance + free RSS is enough. The dashboard runs with a local parquet prediction log at `data/predictions.parquet`.

### Enable persistent storage (recommended)

1. Sign up at [supabase.com](https://supabase.com) (free tier is plenty — 500MB DB, no card)
2. In Supabase SQL Editor, run **all three** migrations in order:
   - [`migrations/001_predictions.sql`](migrations/001_predictions.sql) — track-record table for the AI panels' calibration loop
   - [`migrations/002_searched_tickers.sql`](migrations/002_searched_tickers.sql) — autocomplete cache for non-SEC tickers (crypto, indices, forex, foreign ADRs)
   - [`migrations/003_watchlist.sql`](migrations/003_watchlist.sql) — user-curated watchlist for the Scanner view
3. Grab your Project URL + publishable/anon key from Project Settings → API
4. Add to `.streamlit/secrets.toml` (or Streamlit Cloud app secrets):

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "sb_publishable_..."
   ```

Without Supabase the app still works — predictions and remembered tickers fall back to local files (`data/predictions.parquet` and `data/searched_tickers.json`). The catch is that Streamlit Cloud's filesystem is ephemeral, so both reset on every redeploy.

### Run tests

```bash
pip install -r requirements.txt pytest
pytest tests/
```

See [tests/README.md](tests/README.md) for what's covered and how to add new tests.

---

## Honest limitations

- **Single-ticker ML**, 1 year of daily history ≈ 252 samples. Don't expect edge at 3m horizon — that's why 3m is a regime classifier, not a price target. 1w realistically tops out around 55–60% on liquid names; 1m around 52–55%.
- **Options aggressor is an end-of-day heuristic**, not real trade prints. Expect ~55–65% directional accuracy on liquid strikes. For real buy/sell attribution use Polygon.io ($29/mo) or Unusual Whales ($48+/mo).
- **Calibration needs ≥ 5 resolved predictions per horizon** to kick in. First useful 1w data lands ~5 trading days after your first analysis run (and that's per-horizon, so 3m won't have a real calibration signal for ~3 months).
- **No real-time data**. yfinance is 15-minute delayed and end-of-day accurate for most fields. Not suitable for intraday trading.
- **Paywalled sources we don't directly hit**: Seeking Alpha Pro, Benzinga Pro, Barron's, Bloomberg, WSJ. We do surface their public headlines via aggregators.

---

## Not investment advice

Everything. This is a research tool — a compact dashboard over free public data with some machine learning on top. The calibration panel makes it honest about how often the model has been right, not authoritative about what happens next. Make your own decisions.
