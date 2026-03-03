"""
Unit tests for the performance tracker (WeeklyReview / MonthlyReview).
Uses synthetic trade records — no real market data or LLM calls needed.
"""
from __future__ import annotations

import time

import pytest

from performance.feedback import _resolve_compounding_mode, _strategy_risk_multiplier
from performance.tracker import (
    WeeklyReview,
    _classify_health_tier,
    compute_monthly_review,
    compute_per_strategy_reviews,
    compute_weekly_review,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_week() -> str:
    """Return the ISO week string for the current moment."""
    from datetime import datetime, timezone
    now = datetime.now(tz=timezone.utc)
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def _exit_trade(pnl: float, r_multiple: float = 0.0, ts_ms: int | None = None) -> dict:
    """Build a minimal exit trade dict compatible with DryRunBroker output."""
    return {
        "type":       "exit",
        "pnl":        pnl,
        "r_multiple": r_multiple,
        "timestamp":  ts_ms if ts_ms is not None else int(time.time() * 1000),
    }


def _make_weekly(pnl_pct: float, max_dd: float = 0.0) -> WeeklyReview:
    return WeeklyReview(
        week="2026-W01",
        pnl_usd=pnl_pct * 10_000,
        pnl_pct=pnl_pct,
        win_rate=0.5,
        avg_r_multiple=1.0,
        max_drawdown_pct=max_dd,
        trade_count=5,
    )


# ── WeeklyReview tests ────────────────────────────────────────────────────────

def test_weekly_review_win_rate():
    """3 winning + 2 losing trades → win_rate = 0.60."""
    week = _now_week()
    trades = [
        _exit_trade(pnl=+100.0),
        _exit_trade(pnl=+150.0),
        _exit_trade(pnl=+80.0),
        _exit_trade(pnl=-50.0),
        _exit_trade(pnl=-30.0),
    ]
    result = compute_weekly_review(trades, week)
    assert result.trade_count == 5
    assert abs(result.win_rate - 0.60) < 1e-9


def test_weekly_review_pnl():
    """P&L sum and percentage must be computed correctly."""
    week = _now_week()
    trades = [
        _exit_trade(pnl=+200.0),
        _exit_trade(pnl=-100.0),
        _exit_trade(pnl=+50.0),
    ]
    result = compute_weekly_review(trades, week, starting_equity=10_000.0)
    assert abs(result.pnl_usd - 150.0) < 1e-9
    assert abs(result.pnl_pct - 0.015) < 1e-9   # 150 / 10_000


def test_weekly_review_empty_week():
    """A week with no trades must return zero metrics."""
    result = compute_weekly_review([], "2026-W99")
    assert result.trade_count == 0
    assert result.pnl_usd == 0.0
    assert result.win_rate == 0.0


def test_weekly_review_filters_by_week():
    """Only exit trades in the target week should be counted."""
    week = _now_week()
    # One trade in the right week (now), one with a very old timestamp (different week)
    trades = [
        _exit_trade(pnl=+100.0),                    # current week
        _exit_trade(pnl=+500.0, ts_ms=1_000_000),   # epoch Jan 1 1970 — different week
    ]
    result = compute_weekly_review(trades, week)
    assert result.trade_count == 1
    assert abs(result.pnl_usd - 100.0) < 1e-9


# ── MonthlyReview tests ───────────────────────────────────────────────────────

def test_monthly_review_risk_down_on_drawdown():
    """Monthly max drawdown > 15% must set risk_adjustment = 0.5."""
    weekly_reviews = [_make_weekly(pnl_pct=0.02, max_dd=0.20)]  # 20% drawdown
    result = compute_monthly_review(weekly_reviews, backtest_sharpe=1.5)
    assert result.risk_adjustment == pytest.approx(0.5)
    assert result.max_drawdown_pct == pytest.approx(0.20)


def test_monthly_review_retire_flag():
    """Three consecutive losing weeks must set retire = True."""
    weekly_reviews = [
        _make_weekly(pnl_pct=-0.02),
        _make_weekly(pnl_pct=-0.03),
        _make_weekly(pnl_pct=-0.01),
    ]
    result = compute_monthly_review(weekly_reviews, backtest_sharpe=1.5)
    assert result.consecutive_losing_weeks == 3
    assert result.retire is True


def test_monthly_review_risk_up_on_performance():
    """Monthly PnL > 15% must set risk_adjustment = 1.25."""
    weekly_reviews = [
        _make_weekly(pnl_pct=0.06),
        _make_weekly(pnl_pct=0.05),
        _make_weekly(pnl_pct=0.05),
    ]
    result = compute_monthly_review(weekly_reviews, backtest_sharpe=1.5)
    assert result.pnl_pct == pytest.approx(0.16)    # 6+5+5 = 16%
    assert result.risk_adjustment == pytest.approx(1.25)


def test_monthly_review_risk_neutral():
    """Moderate positive returns must leave risk_adjustment at 1.0."""
    weekly_reviews = [
        _make_weekly(pnl_pct=0.02),
        _make_weekly(pnl_pct=0.03),
    ]
    result = compute_monthly_review(weekly_reviews, backtest_sharpe=1.5)
    assert result.risk_adjustment == pytest.approx(1.0)
    assert result.retire is False


def test_monthly_review_sharpe_decay():
    """sharpe_decay = live_sharpe - backtest_sharpe (can be negative)."""
    weekly_reviews = [
        _make_weekly(pnl_pct=0.01),
        _make_weekly(pnl_pct=0.02),
        _make_weekly(pnl_pct=0.015),
    ]
    result = compute_monthly_review(weekly_reviews, backtest_sharpe=3.0)
    assert result.backtest_sharpe == pytest.approx(3.0)
    # live_sharpe likely < backtest_sharpe → decay is negative
    assert result.sharpe_decay == pytest.approx(result.live_sharpe - 3.0, abs=1e-6)


# ── Health tier classification tests ──────────────────────────────────────────

def test_health_tier_good():
    """All metrics within tolerance → Good."""
    tier = _classify_health_tier(ev_drift_r=0.0, win_rate_drift=0.0, live_dd_vs_mc_ratio=0.5)
    assert tier == "Good"


def test_health_tier_good_at_boundary():
    """Exactly at the Good boundary values → still Good."""
    tier = _classify_health_tier(ev_drift_r=-0.10, win_rate_drift=-0.05, live_dd_vs_mc_ratio=1.0)
    assert tier == "Good"


def test_health_tier_marginal():
    """EV drift between -0.20 and -0.10 → Marginal."""
    tier = _classify_health_tier(ev_drift_r=-0.15, win_rate_drift=0.0, live_dd_vs_mc_ratio=0.0)
    assert tier == "Marginal"


def test_health_tier_poor_ev_drift():
    """EV drift below -0.20 → Poor."""
    tier = _classify_health_tier(ev_drift_r=-0.25, win_rate_drift=0.0, live_dd_vs_mc_ratio=0.0)
    assert tier == "Poor"


def test_health_tier_poor_win_rate():
    """Win rate drift below -0.10 → Poor."""
    tier = _classify_health_tier(ev_drift_r=0.0, win_rate_drift=-0.15, live_dd_vs_mc_ratio=0.0)
    assert tier == "Poor"


def test_health_tier_poor_dd_exceeded():
    """Live DD ratio above 1.2 → Poor."""
    tier = _classify_health_tier(ev_drift_r=0.0, win_rate_drift=0.0, live_dd_vs_mc_ratio=1.3)
    assert tier == "Poor"


def test_monthly_derives_health_tier_from_weeklies():
    """If any weekly is Poor, the monthly health_tier must be Poor."""
    weeklies = [
        _make_weekly(pnl_pct=0.02),
        _make_weekly(pnl_pct=0.01),
    ]
    # Manually override the first weekly's health_tier to Poor
    weeklies[0].health_tier = "Poor"
    result = compute_monthly_review(weeklies, backtest_sharpe=1.5)
    assert result.health_tier == "Poor"


def test_monthly_derives_health_marginal_over_good():
    """If any weekly is Marginal (but none Poor), monthly health_tier is Marginal."""
    weeklies = [
        _make_weekly(pnl_pct=0.02),
        _make_weekly(pnl_pct=0.01),
    ]
    weeklies[1].health_tier = "Marginal"
    result = compute_monthly_review(weeklies, backtest_sharpe=1.5)
    assert result.health_tier == "Marginal"


def test_weekly_review_computes_health_tier():
    """compute_weekly_review should set health_tier field on returned review."""
    week = _now_week()
    trades = [_exit_trade(pnl=+100.0, r_multiple=0.5)]
    result = compute_weekly_review(
        trades, week,
        projected_avg_r=0.6,   # live avg_r (0.5) lower by 0.1R — still Good
        projected_win_rate=0.5,
        mc_p95_dd=0.10,
    )
    assert result.health_tier in ("Good", "Marginal", "Poor")
    assert result.ev_drift_r == pytest.approx(0.5 - 0.6, abs=1e-6)


def test_compute_per_strategy_reviews_groups_by_strategy():
    """compute_per_strategy_reviews must split trades by strategy_id."""
    week = _now_week()
    ts = int(time.time() * 1000)
    trades = [
        {"type": "exit", "strategy_id": "s1", "pnl": 100.0, "r_multiple": 1.0, "timestamp": ts},
        {"type": "exit", "strategy_id": "s1", "pnl":  50.0, "r_multiple": 0.5, "timestamp": ts},
        {"type": "exit", "strategy_id": "s2", "pnl": -30.0, "r_multiple": -0.3, "timestamp": ts},
    ]
    reviews = compute_per_strategy_reviews(
        trades, week, starting_equity=10_000.0,
        projected_avg_r=0.5, projected_win_rate=0.5,
        mc_p95_dd_by_strategy={},
    )
    assert set(reviews.keys()) == {"s1", "s2"}
    assert reviews["s1"].trade_count == 2
    assert reviews["s2"].trade_count == 1


# ── Compounding mode unlock tests ─────────────────────────────────────────────

def test_compounding_unlocks_monthly_at_3_months():
    """After 3 months at Good tier, mode upgrades from 'none' to 'monthly'."""
    mode = _resolve_compounding_mode("s1", tier="Good", months_good=3, current_mode="none")
    assert mode == "monthly"


def test_compounding_unlocks_per_trade_at_6_months():
    """After 6 months at Good tier, mode upgrades to 'per_trade'."""
    mode = _resolve_compounding_mode("s1", tier="Good", months_good=6, current_mode="monthly")
    assert mode == "per_trade"


def test_compounding_no_upgrade_when_marginal():
    """Marginal tier must not trigger a compounding upgrade."""
    mode = _resolve_compounding_mode("s1", tier="Marginal", months_good=5, current_mode="none")
    assert mode == "none"


def test_compounding_no_upgrade_when_poor():
    """Poor tier must not trigger a compounding upgrade."""
    mode = _resolve_compounding_mode("s1", tier="Poor", months_good=10, current_mode="none")
    assert mode == "none"


# ── Per-strategy risk multiplier tests ────────────────────────────────────────

def test_per_strategy_risk_multiplier_poor():
    """Poor health tier → 0.5x multiplier."""
    from performance.tracker import MonthlyReview as MR
    monthly = MR(
        month="2026-01", pnl_pct=0.02, live_sharpe=1.0, backtest_sharpe=1.5,
        sharpe_decay=-0.5, max_drawdown_pct=0.05, consecutive_losing_weeks=0,
        risk_adjustment=1.0, retire=False,
    )
    mult = _strategy_risk_multiplier("Poor", monthly)
    assert mult == pytest.approx(0.5)


def test_per_strategy_risk_multiplier_marginal():
    """Marginal health tier → 0.75x multiplier regardless of monthly P&L."""
    from performance.tracker import MonthlyReview as MR
    monthly = MR(
        month="2026-01", pnl_pct=0.20, live_sharpe=2.0, backtest_sharpe=1.5,
        sharpe_decay=0.5, max_drawdown_pct=0.03, consecutive_losing_weeks=0,
        risk_adjustment=1.25, retire=False,
    )
    mult = _strategy_risk_multiplier("Marginal", monthly)
    assert mult == pytest.approx(0.75)


def test_per_strategy_risk_multiplier_good_uses_monthly():
    """Good tier defers to monthly risk_adjustment (inherits 1.25x on strong month)."""
    from performance.tracker import MonthlyReview as MR
    monthly = MR(
        month="2026-01", pnl_pct=0.20, live_sharpe=2.0, backtest_sharpe=1.5,
        sharpe_decay=0.5, max_drawdown_pct=0.03, consecutive_losing_weeks=0,
        risk_adjustment=1.25, retire=False,
    )
    mult = _strategy_risk_multiplier("Good", monthly)
    assert mult == pytest.approx(1.25)
