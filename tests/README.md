# Testing StockIQ

Fast, no-network, pure-logic tests for the bits of the app that carry real risk of silent regressions. Run time is usually < 2 seconds for the whole suite.

---

## Running the tests

```bash
# from the repo root
pip install -r requirements.txt pytest
pytest tests/                 # all tests
pytest tests/test_calibration.py -v    # one file, verbose
pytest -k "cluster"            # tests matching a substring
pytest --ff                    # run failed-first on next invocation
```

No API keys, no network, no Supabase required for any test.

---

## What's covered

| Test file | What it tests | Why it matters |
|---|---|---|
| `test_imports.py` | Every `stockiq.*` module imports cleanly + backward-compat shims resolve | Catches circular imports and breaking refactors |
| `test_deprecations.py` | No `fillna(method=)` or `use_container_width` anywhere in `stockiq/` | pandas 2.2 + Streamlit 1.45 both removed these; a regression would crash prod |
| `test_prediction_log.py` | `PredictionLog` round-trip (log / read), empty-summary shape, aggregation of pre-resolved rows into hit rate + calibration buckets | If this breaks, the Track Record panel silently corrupts |
| `test_calibration.py` | `calibrate_probs` and `calibrate_confidence` math across hit-rate regimes (too little data, 50%, 75%, 100%, anti-signal) + probability renormalization | This is the feedback loop that shrinks displayed confidence; a math error silently breaks every AI panel |
| `test_options_heuristics.py` | Bid/ask aggressor classifier (BUY / SELL / MID / —) across edge cases (locked market, no spread, stale last, zero bid) + cluster detection (three nearby strikes, same side + expiry, within 10%) | These heuristics drive the B/S/M tags in the Unusual Options panel |
| `test_sentiment.py` | Keyword sentiment classifier — polarity, case-insensitivity, word-boundary matching (no false "win" match in "winter"), bounded [-1, 1] | Feeds multiple ML features and the news feed per-article colors |
| `test_formatters.py` | `fmt_price` / `fmt_pct` / `fmt_ratio` / `fmt_pct_ratio` / `fmt_big_money` / `_cls` edge cases (None, zero, negative, large) | Every panel uses these; a regression silently breaks every number display |
| `test_earnings_calendar.py` | `get_earnings_calendar` parses both the new dict-form and old DataFrame-form `ticker.calendar`, falls back to `earnings_dates`, handles exceptions safely, ignores past dates | yfinance quietly changed this return type; parsing bugs here caused "NEXT EARNINGS —" everywhere |

---

## What's *not* covered (intentionally)

- **Live yfinance calls** — too slow, too flaky for CI. The `test_earnings_calendar.py` tests stub the Ticker.
- **Streamlit UI rendering** — headless pixel-diff is overkill for a dashboard. Manual visual verification on `streamlit run sidebar_web.py` is faster.
- **Supabase REST round-trips** — requires a running Supabase project. Integration-test scope, not unit-test.
- **ML training** — needs a real OHLCV DataFrame + ~250 trading days. Could be added with a cached CSV fixture; not done yet.

---

## Adding a test

New unit tests go in `tests/test_<area>.py`. Conventions:

1. **No network**. Stub `self.stock` on analyzer tests via `UniversalStockAnalyzer.__new__(cls)` + manual attribute assignment (see `test_earnings_calendar.py` for the pattern).
2. **No Streamlit**. Test the pure helper functions. Panel render functions emit HTML fragments — if you need to test one, assert on the output string.
3. **`tmp_path` fixture** for anything that writes to disk (the prediction log does).
4. **Keep each test one concern**. `test_probs_random_hit_rate_full_shrinkage` is named that way on purpose; a failure message tells you exactly which behavior broke.

Pull the test into the pattern table above when you add it so the coverage inventory stays accurate.

---

## Local dev loop

Quick checklist before pushing:

```bash
# 1. Syntax / import regression
python -c "import sidebar_web, stockiq"

# 2. All unit tests
pytest tests/

# 3. Full UI boot (requires Streamlit 1.45+)
streamlit run sidebar_web.py
# browse to http://localhost:8501 and pick a few tickers
```

If a test fails that you *meant* to change, update the test in the same commit — don't leave a failing test in the tree.
