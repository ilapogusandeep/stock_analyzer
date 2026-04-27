"""Composite "unusualness" score for ranking scanner rows.

Combines four signals into a single 0..100 score so we can sort the
universe and surface the top-N candidates. All component scores are
clamped to [0, 25] so no single dimension can dominate.

Scoring components (each 0..25):

    score_options   25 * min(1, unusual_count / 5)
        — 5+ unusual strikes = full score
    score_aggressor 25 * min(1, |aggressor_net| / 5)
        — |B-S| of 5+ = full score (sign preserved separately as bias)
    score_news      25 * min(1, news_velocity / 3)
        — 3x today vs 7d avg = full score
    score_move      25 * min(1, |change_1d| / 0.05)
        — 5%+ daily move = full score

Total = score_options + score_aggressor + score_news + score_move
"""

from __future__ import annotations

from typing import Any


def _clip_unit(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def score_signal(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compute a composite unusualness score (0..100) for one signal
    snapshot. Mutates and returns the snapshot with extra keys:

        score, score_options, score_aggressor, score_news, score_move,
        bias  ('BULLISH' / 'BEARISH' / 'NEUTRAL' from aggressor net + 1d return)
    """
    unusual = snapshot.get("unusual_count") or 0
    agg = snapshot.get("aggressor_net") or 0
    velocity = float(snapshot.get("news_velocity") or 0)
    change = float(snapshot.get("change_1d") or 0)

    score_options = 25.0 * _clip_unit(unusual / 5.0)
    score_aggressor = 25.0 * _clip_unit(abs(agg) / 5.0)
    score_news = 25.0 * _clip_unit(velocity / 3.0)
    score_move = 25.0 * _clip_unit(abs(change) / 0.05)

    total = score_options + score_aggressor + score_news + score_move

    # Bias: combine aggressor sign with day return sign. If both agree
    # we have a directional bet; if they disagree, neutral.
    if agg > 0 and change > 0:
        bias = "BULLISH"
    elif agg < 0 and change < 0:
        bias = "BEARISH"
    elif agg > 1 and change > -0.005:
        bias = "BULLISH"
    elif agg < -1 and change < 0.005:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    snapshot.update({
        "score": round(total, 1),
        "score_options": round(score_options, 1),
        "score_aggressor": round(score_aggressor, 1),
        "score_news": round(score_news, 1),
        "score_move": round(score_move, 1),
        "bias": bias,
    })
    return snapshot


def rank_signals(snapshots: list[dict]) -> list[dict]:
    """Apply score_signal to each snapshot and return them sorted by
    descending score."""
    scored = [score_signal(s) for s in snapshots]
    return sorted(scored, key=lambda r: r.get("score", 0), reverse=True)
