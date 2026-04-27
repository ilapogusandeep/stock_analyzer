# StockIQ — Architecture

This doc is for contributors. User-facing features are in [README.md](README.md).

---

## High-level flow

```mermaid
flowchart LR
    classDef external fill:#1e293b,stroke:#475569,color:#e2e8f0
    classDef ui fill:#1e3a8a,stroke:#3b82f6,color:#dbeafe
    classDef core fill:#3b0764,stroke:#a855f7,color:#e9d5ff
    classDef data fill:#064e3b,stroke:#10b981,color:#d1fae5
    classDef store fill:#422006,stroke:#f59e0b,color:#fef3c7
    classDef passthrough fill:#0f172a,stroke:#334155,color:#94a3b8,stroke-dasharray:3 3

    User((User browser)):::ui
    Streamlit[Streamlit Cloud<br/>sidebar_web.py]:::ui
    Analyzer[UniversalStockAnalyzer<br/>stockiq.core.analyzer]:::core

    Collector[EnhancedDataCollector<br/>stockiq.data.collector]:::data
    Inst[EnhancedInstitutionalData<br/>stockiq.data.institutional]:::data
    Options[Options module<br/>stockiq.data.options]:::data
    Tickers[Ticker universe<br/>POPULAR + SEC + searched<br/>stockiq.data.tickers]:::data

    ML[ML pipeline<br/>RF + GB ensemble<br/>5d · 21d · 63d horizons]:::core
    PredLog[PredictionLog<br/>stockiq.core.prediction_log]:::core

    Components[UI components<br/>stockiq.ui.components]:::ui
    Theme[Dark theme CSS<br/>stockiq.ui.theme]:::ui

    YFPrices[yfinance core<br/>prices · info · options · 13F · earnings]:::external
    YFNews[yfinance.ticker.news<br/>aggregator]:::external
    GNews[Google News RSS<br/>aggregator]:::external
    YahooRSS[Yahoo Finance RSS<br/>aggregator]:::external
    MWRSS[MarketWatch RSS<br/>aggregator]:::external
    SectorETF[Sector SPDRs<br/>XLK/XLF/XLV/...]:::external
    SECList[SEC company_tickers.json<br/>~10K US equities]:::external
    Supabase[(Supabase<br/>predictions + searched_tickers tables)]:::store
    Parquet[(data/predictions.parquet<br/>local fallback)]:::store
    SearchedFile[(data/searched_tickers.json<br/>local fallback)]:::store

    Publishers[Publisher labels passed through:<br/>Motley Fool · Benzinga · IBD · MarketBeat<br/>Seeking Alpha · CNBC · Reuters · Zacks<br/>Barron's · Business Insider · MSN · TIKR<br/>StockStory · TradingKey · ...]:::passthrough

    User -->|HTTPS| Streamlit
    Streamlit --> Analyzer
    Streamlit --> PredLog
    Streamlit --> Components
    Streamlit --> Theme
    Streamlit --> Tickers
    Tickers -->|7-day cached fetch| SECList
    Tickers -.->|if SUPABASE_URL set| Supabase
    Tickers -.->|fallback| SearchedFile

    Analyzer --> Collector
    Analyzer --> Inst
    Analyzer --> ML
    Analyzer -->|sector lookup| SectorETF

    Collector --> YFNews
    Collector --> GNews
    Collector --> YahooRSS
    Collector --> MWRSS
    Inst --> YFPrices
    Options --> YFPrices

    YFNews -.->|publisher displayName| Publishers
    GNews -.->|<source> tag per item| Publishers
    YahooRSS -.->|item title| Publishers
    MWRSS -.->|item title| Publishers
    Publishers -.->|labels rendered in News Feed| Components

    ML --> Analyzer

    PredLog -.->|if SUPABASE_URL set| Supabase
    PredLog -.->|fallback| Parquet
    PredLog -->|resolve prices| YFPrices

    Components --> User
```

Legend: green = data layer, purple = core logic, blue = UI, orange = storage, slate = external services we **call directly**, dashed grey = publisher names that appear in the News Feed but are **passed through** the aggregators above (we never make HTTP calls to those outlets directly).

---

## Package layout

```
stock_analyzer/
├── sidebar_web.py              # Streamlit entry point (the whole page render)
├── stockiq/
│   ├── core/
│   │   ├── analyzer.py         # UniversalStockAnalyzer — orchestrates the full analyze() pipeline
│   │   └── prediction_log.py   # PredictionLog + calibrate_probs() / calibrate_confidence()
│   ├── data/
│   │   ├── collector.py        # EnhancedDataCollector — news, social, analyst, options data aggregation
│   │   ├── institutional.py    # EnhancedInstitutionalData — 13F holders, insider transactions, earnings
│   │   ├── options.py          # get_options_flow() + get_unusual_activity()
│   │   └── tickers.py          # POPULAR_TICKERS dict for the autocomplete dropdown
│   ├── models/
│   │   ├── predictor.py        # EnhancedPricePredictor (older, kept for back-compat shims)
│   │   └── sentiment.py        # AdvancedSentimentAnalyzer (VADER/FinBERT placeholder)
│   └── ui/
│       ├── components.py       # All render functions: header_band, kv_block, news_feed_block, ...
│       └── theme.py            # Dark CSS injected once per Streamlit session
├── migrations/
│   └── 001_predictions.sql     # Supabase schema
├── tests/
│   ├── test_imports.py         # Package smoke test
│   ├── test_deprecations.py    # Lint: no deprecated Streamlit/pandas APIs
│   ├── test_prediction_log.py  # PredictionLog round-trip + summary
│   ├── test_calibration.py     # calibrate_probs / calibrate_confidence math
│   ├── test_options_heuristics.py  # Aggressor + cluster detection
│   ├── test_sentiment.py       # Keyword-based _analyze_text_sentiment
│   ├── test_formatters.py      # fmt_price / fmt_pct / fmt_big_money
│   └── test_earnings_calendar.py   # dict vs DataFrame parsing
└── archive/                    # Older experimental files (streamlit_app.py etc.), not imported
```

At the repo root there are also backward-compat shim modules (`enhanced_data_collector.py`, `enhanced_institutional_data.py`, etc.) that just do `from stockiq.data.X import *`. They exist so that older user scripts importing those top-level names keep working; new code should import from `stockiq.*`.

---

## Data flow per analysis

Triggered by selecting a ticker (auto-analyze) or clicking the Analyze button:

1. **`sidebar_web.py`** reads the ticker from the selectbox, calls `UniversalStockAnalyzer(ticker).analyze_stock()`.
2. **Analyzer** orchestrates fetches in this order:
   - `EnhancedDataCollector.collect_enhanced_data()` → news sentiment (7 sources), social, analyst, options sentiment, economic indicators, rankings.
   - `EnhancedInstitutionalData.get_comprehensive_institutional_data()` → top 13F holders, insider transactions, earnings history, analyst recommendations.
   - `stock.history(period="1y")` → 1-year daily OHLCV.
   - `stock.info` → fundamentals, metadata.
   - `get_earnings_calendar()` → next earnings date (dict-form calendar + earnings_dates fallback).
   - `_fetch_sector_etf_history(info)` → 1-year sector ETF prices for peer relative strength features.
3. **Derived analyses**: `compute_technical_indicators(hist)`, `analyze_fundamentals(info, hist)`, `_process_enhanced_sentiment(enhanced_data)`.
4. **ML ensemble** runs three times with different horizons:
   - `create_enhanced_ml_prediction(..., horizon_days=5)` — 1-week direction
   - `create_enhanced_ml_prediction(..., horizon_days=21)` — 1-month direction
   - `create_regime_prediction(..., horizon_days=63)` — 3-month 3-class regime
5. **Backtest** runs `enhanced_backtest_strategy()` for the SPY + buy-and-hold comparison (backend only; not rendered).
6. **sidebar_web.py** receives the combined result dict, calls each panel render function against it, logs the three predictions to Supabase, and lazily resolves any predictions older than their horizon.

---

## ML pipeline

### Feature engineering

`_create_enhanced_features()` builds a `pd.DataFrame` indexed by trading day, with ~35 columns. The full list lives in the README; notable ones:

- **Rolling** (Series): returns at multiple windows, RSI, MACD, BB, SMA ratios, volume ratio, peer/sector relative strength at 1m and 3m.
- **Static** (broadcast across rows): news sentiment aggregates, 13F net flow, analyst ratings, economic data, days-to-earnings. These look identical for every row; they represent the "now" snapshot that informs the most recent prediction. Not ideal for training but the alternative (fetching historical news sentiment per day) is out of scope for free-tier.

### Training

- **Train/test split**: chronological 80/20, **no shuffle**. A shuffled split leaks future info into the training set and is the single most common mistake in time-series ML. Reported accuracy is genuine walk-forward performance.
- **Ensemble**: `RandomForestClassifier(n_estimators=100)` (60% weight) + `GradientBoostingClassifier(n_estimators=100)` (40%). Probabilities blended linearly.
- **Binary horizons** (1w, 1m): label = `Close[t+H] > Close[t]`. 3-class neutral derived post-hoc as `1 - |bull - bear|` and renormalized.
- **Regime classifier** (3m): label = `{BEARISH, SIDEWAYS, BULLISH}` based on threshold `σ · √63 · 0.5`. `class_weight='balanced'` to prevent the SIDEWAYS majority from dominating.
- **Price targets**: scale with horizon via `σ · √H · (0.5 + p_direction)`, not hard-coded percentages. A 1-week bullish target sits ~1σ above spot; a 1-month target ~2σ.

### Calibration (the feedback loop)

After each analysis, `_pred_log.log(...)` writes three rows to Supabase — one per horizon. On the next render, `PredictionLog.summary(horizon_days=H)` is called for each of 5/21/63 to pull that horizon's historical hit rate. If ≥ 5 resolved predictions exist, `calibrate_probs()` shrinks the raw model output toward uniform by a factor scaled to demonstrated edge:

```
trust = min(1.0, 2 · |hit_rate − 0.5|)
p_cal = p_raw + (⅓ − p_raw) · (1 − trust)
```

At hit_rate = 0.5, trust = 0 → full shrinkage to uniform. At 1.0, trust = 1 → no shrinkage. Below 0.5 we still shrink (small bad samples are noise more often than true inverse signal).

---

## Storage layer

Two persisted-state surfaces — predictions and searched tickers. Each one chooses Supabase vs a local file at runtime based on whether `SUPABASE_URL` + `SUPABASE_KEY` are in `st.secrets` or env. Both share the same dispatch pattern: a private `_supabase_*` path and a `_file_*` fallback, with the public function picking the live backend.

### Supabase (preferred)

Two tables, both managed via PostgREST:

| Table | Migration | Purpose | Key ops |
|---|---|---|---|
| `predictions` | `migrations/001_predictions.sql` | Per-analysis prediction rows for the AI calibration loop | `_log_supabase()` (POST), `_resolve_supabase()` (SELECT pending + PATCH), `_read_all_supabase()` (SELECT all) |
| `searched_tickers` | `migrations/002_searched_tickers.sql` | Autocomplete cache for non-SEC tickers (crypto, indices, forex, foreign ADRs) | `_remember_supabase()` (UPSERT via `on_conflict=ticker` + `Prefer: resolution=merge-duplicates`), `_load_searched_supabase()` (SELECT all, ordered by last_seen) |

Both tables disable RLS for the single-user app (the publishable key is safe to commit; Supabase is designed for client-side use). REST traffic uses raw `requests` to keep the cold-start payload small — no `supabase-py` dependency.

A note on the resolution job: PostgREST can't express `timestamp + resolution_horizon_days * interval '1 day' ≤ now()` as a query param, so `_resolve_supabase()` pulls up to 500 unresolved rows and filters by per-row horizon **in Python**, then groups by ticker so each yfinance price call resolves multiple rows.

### Local file fallback

| File | Purpose | Backing function |
|---|---|---|
| `data/predictions.parquet` | Same schema as the predictions table; full read + full rewrite per log call | `_log_parquet`, `_resolve_parquet` |
| `data/searched_tickers.json` | `{ticker: company_name}` flat dict | `_remember_file`, `_load_searched_file` |

On Streamlit Cloud both files live on the ephemeral container filesystem — they survive reruns within a container but reset on redeploy. That's the main reason to prefer Supabase once you're running in production.

### Cost on the Supabase free tier

- ~150 bytes per row in either table; 10K predictions = 1.5MB; 10K remembered tickers = 1.5MB. Both stay well under the 500MB DB allowance for years.
- Per analysis: 3 INSERTs to `predictions` (1w/1m/3m) + 1 UPSERT to `searched_tickers` (skipped for curated tickers) + 3 SELECTs for calibration + 1 SELECT for resolution + 1 SELECT for the dropdown render = ~9 REST calls. The free tier has no per-request cap (only egress: 5GB/month, irrelevant at our payload size).

---

## UI layer

Streamlit renders the page top-to-bottom on every run (no partial updates except within `st.selectbox` reruns). To get a dense, non-scrollable layout we:

1. **Inject custom CSS** once via `inject_theme()` — hides the default Streamlit header/toolbar, crushes top padding, defines every panel's styling.
2. **Build HTML fragments in Python** and emit via `st.markdown(unsafe_allow_html=True)`. Most panels are single-shot HTML; only the chart uses `st.plotly_chart`.
3. **4-column grid** (`0.18 / 0.42 / 0.20 / 0.20`) via `st.columns()`. Each column is a thematic chunk (fundamentals / chart+analyst / smart money / AI+news).
4. **Session state** tracks the last analyzed ticker so a selectbox change auto-triggers a rerun with no button press.

### Hot-reload safety

Streamlit on Python 3.13 occasionally drops subpackages from `sys.modules` between reruns, triggering `KeyError: 'stockiq.core'` on the next import. Two defenses:

1. **`stockiq/__init__.py`** eagerly imports every subpackage (wrapped in `try/except` so a transient failure doesn't poison the parent module).
2. **`sidebar_web.py`** wraps its top-level `from stockiq.* import ...` statements in `_import_stockiq_modules()` which retries up to 3 times: on `KeyError` or `ImportError` starting with `'stockiq'`, it purges every `stockiq*` entry from `sys.modules` and reimports.

UI functions are also written defensively against signature mismatches: e.g., `probability_scenarios_combined` accepts `**kwargs` so new callers passing extra named args won't crash a stale cached copy of the function.

---

## External services

### Endpoints we call directly

| Service | Purpose | Cost | Rate limit | Failure mode |
|---|---|---|---|---|
| **yfinance core** | Prices, info, options chains, holders (13F), earnings calendar, sector lookup | Free | Unofficial — scrapes Yahoo Finance since ~2017, occasionally rate-limits | Silent retry, then empty result |
| **yfinance.ticker.news** (aggregator) | Curated news with real publisher display names | Free | Same as above | Skip news source |
| **Google News RSS** (aggregator) | `news.google.com/rss/search?q={ticker}+stock`. Each `<item>` carries a `<source>` tag with the real outlet name | Free | None officially, practically loose | Timeout → skip source |
| **Yahoo Finance RSS** (aggregator) | `feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}` | Free | Same | Timeout → skip |
| **MarketWatch RSS** (aggregator) | Per-ticker headline feed | Free | Same | Timeout → skip |
| Sector SPDR ETFs | Peer relative strength (XLK / XLF / XLV / XLY / XLP / XLI / XLE / XLC / XLU / XLRE / XLB) — fetched via yfinance | Free (via yfinance) | Inherited | Feature defaults to 0 |
| **SEC company_tickers.json** | All ~10K US-listed equities — fed into the autocomplete dropdown | Free, official | Strict on User-Agent format ("Name email@domain"); rate-limited if abused | Falls back to curated POPULAR_TICKERS only |
| Supabase | Persistent prediction log | Free tier (500MB DB) | 50K monthly active users | Parquet fallback |

### Publishers shown in the News Feed (passed through, not directly fetched)

The publisher labels you see on each headline (Motley Fool, Benzinga, MarketBeat, Seeking Alpha, IBD, CNBC, Reuters, Barron's, Business Insider, Zacks, MSN, TIKR, StockStory, TradingKey, ...) are **propagated through** the aggregators above. We never make an HTTP request to `benzinga.com` or `fool.com` directly.

Where each label comes from:
- **`yfinance.ticker.news`** returns articles from Yahoo's curated network — Motley Fool, IBD, Reuters, Barron's, Business Insider, Bloomberg, Zacks — with `provider.displayName` populated.
- **Google News RSS** carries CNBC, MarketBeat, MSN, TIKR.com, StockStory, TradingKey, Seeking Alpha (when not paywalled), Benzinga (free articles), and many smaller outlets — extracted from the `<source>` tag in each `<item>`.
- **MarketWatch / Yahoo Finance RSS** are MarketWatch- and Yahoo-branded headline feeds; titles surface as-is.

After all sources are fetched, the aggregator de-duplicates by lowercased title prefix and keeps the freshest copy of any headline that appears via multiple aggregators (so a Reuters story from yfinance + the same story from Google News appears once).

### Why we don't hit publishers directly

Most premium outlets (Seeking Alpha Pro, Benzinga Pro, Barron's, Bloomberg, WSJ, FT) are paywalled — there's no free per-ticker RSS for them anymore. Free outlets (CNBC, MarketWatch, MarketBeat, Motley Fool) generally don't expose per-ticker feeds either; they syndicate to aggregators and let downstream consumers (Yahoo, Google News) handle distribution. Adding a direct paid integration would be ~50 LOC of REST glue per service:

| Service | Cost | What it adds |
|---|---|---|
| Polygon.io | $29/mo | Real options time-and-sales (proper buy/sell attribution) |
| Unusual Whales | $48–98/mo | Curated UOA + Congress trades + dark pool prints |
| Benzinga Pro | $99+/mo | Real-time press releases + analyst notes feed |
| Tradier | $10/mo | Real-time options chains + Greeks |

---

## Adding a new feature — checklist

1. **Data fetch**: add a method to `EnhancedDataCollector` (for news/social) or `EnhancedInstitutionalData` (for holdings/earnings), or a standalone function in `stockiq/data/` for something else. Return a plain dict; fail closed with `{}` or `None`.
2. **Feature engineering**: add the extracted fields to `_create_enhanced_features()`. Use a default (0 / 0.5) when upstream data is missing so `.fillna(0)` doesn't silently corrupt a real zero.
3. **UI panel**: write a render function in `stockiq/ui/components.py` that takes the new dict and emits HTML via `panel_open` / `panel_close`. Add the CSS to `stockiq/ui/theme.py`.
4. **Wire into `sidebar_web.py`**: import the new render function, call it in the right column.
5. **Test**: add a unit test under `tests/` covering the pure-logic parts (feature extraction, any math).
6. **Backward-compat shim** (only if naming something that external CLI scripts might import): add a root-level module `my_new_thing.py` that does `from stockiq.x.my_new_thing import *`.

---

## Known tech debt

- **XGBoost mention in older README was aspirational**, not real. The ensemble is RF + GB only.
- **`models/predictor.py` and `models/sentiment.py`** are half-wired. `create_ml_prediction()` in `analyzer.py` calls them as a fallback but the primary path is the inline ML in `create_enhanced_ml_prediction()`.
- **FinBERT is documented as a future option** but not wired. The current sentiment is a hand-curated keyword list in `_analyze_text_sentiment()`.
- **Some static features** (news sentiment aggregates, 13F flow) are broadcast across all training rows. This is a known imperfection — they should really be historical time series, but aggregator APIs don't expose them cheaply.
- **No partial UI updates**: every Streamlit rerun re-fetches yfinance data (yfinance does its own HTTP caching so it's not as bad as it sounds, but a proper cache layer would help).

---

## See also

- [README.md](README.md) — user-facing feature docs
- [tests/README.md](tests/README.md) — test coverage + how to add tests
- [migrations/001_predictions.sql](migrations/001_predictions.sql) — Supabase schema
