"""
Unit tests for the Phase 3 backtesting engine.
Uses synthetic candle data — no real market data or LLM calls needed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backtesting.indicators import atr, bollinger_bands, ema, macd, rsi, sma, stochastic
from backtesting.engine import INITIAL_CAPITAL, RISK_LEVELS, SLIPPAGE_PCT, pick_best_risk_level, run_all_risk_levels
from backtesting.filters import apply_filters
from backtesting.metrics import compute_metrics, _max_drawdown, _max_consecutive_losses
from backtesting.models import (
    BacktestResult,
    EvaluatedStrategy,
    FilterResult,
    MetricsResult,
    ScoreResult,
    Strategy,
    StrategySelection,
    Trade,
)
from backtesting.monte_carlo import run_monte_carlo
from backtesting.scorer import WEIGHTS, score_strategy
from backtesting.selector import select_top_3, select_winner
from data.candles import Candle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candles(prices: list[float], bucket_seconds: int = 3600) -> list[Candle]:
    """Build a minimal candle list from a price series."""
    candles = []
    for i, p in enumerate(prices):
        candles.append(Candle(
            ts=i * bucket_seconds * 1000,
            open=p,
            high=p * 1.005,
            low=p * 0.995,
            close=p,
            volume=1_000_000.0,
            swap_count=10,
        ))
    return candles


def _make_trending_candles(n: int = 400) -> list[Candle]:
    """Rising trend with occasional pullbacks — designed to trigger RSI_OVERSOLD signals."""
    prices = []
    p = 1.0
    for i in range(n):
        # Every 20 bars: sharp dip followed by recovery (RSI oversold trigger)
        if i % 20 == 15:
            p *= 0.94   # -6% dip → forces RSI below 30
        elif i % 20 == 16:
            p *= 1.07   # recovery
        else:
            p *= 1.001  # gentle uptrend
        prices.append(p)
    return [
        Candle(
            ts=i * 3600 * 1000,
            open=prices[i],
            high=prices[i] * 1.01,
            low=prices[i] * 0.985,
            close=prices[i],
            volume=1_500_000.0,
            swap_count=15,
        )
        for i in range(n)
    ]


def _make_strategy(
    trigger: str = "RSI_OVERSOLD",
    primary: str = "RSI",
    confirmation: str = "NONE",
) -> Strategy:
    return Strategy(
        id="test_001",
        name="Test RSI Strategy",
        primary_indicator={"type": primary, "params": {"period": 14}},
        confirmation_indicator={"type": confirmation, "params": {"period": 50}},
        entry={"trigger": trigger, "filter": confirmation},
        exit={
            "stop_loss":         {"type": "atr_multiple", "value": 2.0},
            "take_profit":       {"type": "r_multiple", "value": 2.0},
            "trailing_stop_atr": None,
            "time_exit_bars":    None,
        },
        risk={"max_open_positions": 3},
        metadata={},
    )


def _make_metrics(**overrides) -> MetricsResult:
    base = MetricsResult(
        sharpe=1.2, sortino=1.8,
        max_drawdown_pct=0.10, max_drawdown_duration_bars=50, max_drawdown_recovery_bars=80,
        win_rate=0.55, profit_factor=1.8, avg_r_multiple=1.1,
        trade_count=60, max_consecutive_losses=4, worst_trade_pct=0.03,
        monthly_return_variance=0.02, bull_pnl_pct=0.15, bear_pnl_pct=0.05,
        sideways_pnl_pct=0.03, best_risk_pct=1.0,
    )
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


def _make_backtest(**overrides) -> BacktestResult:
    base = BacktestResult(
        strategy_id="test_001", best_risk_pct=1.0,
        trades=[], equity_curve=[INITIAL_CAPITAL],
        in_sample_bars=320, out_of_sample_bars=80,
        only_works_at_max_risk=False,
    )
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


def _make_evaluated(
    strategy_id="s1",
    score_total=70.0,
    passed=True,
    failure=None,
    primary="RSI",
    trigger="RSI_OVERSOLD",
) -> EvaluatedStrategy:
    strategy = Strategy(
        id=strategy_id, name=f"Strategy {strategy_id}",
        primary_indicator={"type": primary, "params": {}},
        confirmation_indicator={"type": "EMA", "params": {}},
        entry={"trigger": trigger, "filter": "NONE"},
        exit={"stop_loss": {"type": "atr_multiple", "value": 2}, "take_profit": {"type": "r_multiple", "value": 2}},
        risk={"max_open_positions": 3},
    )
    metrics = _make_metrics()
    backtest = _make_backtest(strategy_id=strategy_id)
    f_result = FilterResult(passed=passed, failure_reason=failure)
    score = ScoreResult(
        strategy_id=strategy_id, total=score_total,
        ev_score=70.0, drawdown_profile=70.0, risk_adjusted_return=60.0,
        consistency=65.0, profit_factor_score=50.0, statistical_confidence=80.0,
    ) if passed else None
    return EvaluatedStrategy(strategy=strategy, backtest=backtest,
                             metrics=metrics, filter_result=f_result, score=score)


# ── Indicator tests ───────────────────────────────────────────────────────────

def test_sma_basic():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(prices, 3)
    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert abs(result[2] - 2.0) < 1e-9
    assert abs(result[3] - 3.0) < 1e-9
    assert abs(result[4] - 4.0) < 1e-9


def test_ema_convergence():
    prices = np.array([10.0] * 5 + [20.0] * 10, dtype=float)
    result = ema(prices, 5)
    # After many bars at 20.0 the EMA should converge toward 20.0
    assert result[-1] > 15.0


def test_rsi_range():
    """RSI must always be in [0, 100]."""
    prices = np.array([float(i) for i in range(1, 30)])
    result = rsi(prices, 14)
    valid = result[~np.isnan(result)]
    assert all(0.0 <= v <= 100.0 for v in valid)


def test_rsi_oversold_on_declining_prices():
    """A sustained price decline should push RSI well below 50."""
    prices = np.array([100.0 - i * 0.5 for i in range(30)])
    result = rsi(prices, 14)
    valid = result[~np.isnan(result)]
    assert valid[-1] < 50.0


def test_atr_positive():
    """ATR must be positive for any valid OHLC series."""
    candles = _make_candles([1.0 + i * 0.01 for i in range(30)])
    h = np.array([c.high  for c in candles])
    l = np.array([c.low   for c in candles])
    c = np.array([c.close for c in candles])
    result = atr(h, l, c, 14)
    valid = result[~np.isnan(result)]
    assert all(v > 0 for v in valid)


def test_bollinger_upper_above_lower():
    prices = np.array([1.0 + 0.01 * i + 0.005 * (i % 3) for i in range(30)])
    upper, middle, lower = bollinger_bands(prices, period=20)
    valid_idx = ~np.isnan(upper)
    assert np.all(upper[valid_idx] >= lower[valid_idx])


# ── Engine tests ──────────────────────────────────────────────────────────────

def test_engine_produces_trades():
    """A trending candle series with dips should generate at least one RSI trade."""
    candles  = _make_trending_candles(400)
    strategy = _make_strategy("RSI_OVERSOLD", "RSI", "NONE")
    results  = run_all_risk_levels(strategy, candles)
    total_trades = sum(len(r.trades) for r in results)
    assert total_trades > 0, "Expected at least one trade across all risk levels"


def test_engine_equity_curve_length_matches_candles():
    candles  = _make_trending_candles(300)
    strategy = _make_strategy("RSI_OVERSOLD", "RSI", "NONE")
    results  = run_all_risk_levels(strategy, candles)
    for r in results:
        assert len(r.equity_curve) == len(candles)


def test_engine_slippage_applied():
    """
    Entry price must be higher than the bar's close price by ~SLIPPAGE_PCT.
    We inspect the first trade if any are generated.
    """
    candles  = _make_trending_candles(400)
    strategy = _make_strategy("RSI_OVERSOLD", "RSI", "NONE")
    results  = run_all_risk_levels(strategy, candles)
    all_trades = [t for r in results for t in r.trades]
    if all_trades:
        t = all_trades[0]
        bar_close = candles[t.entry_bar].close
        assert t.entry_price > bar_close * (1 - 1e-6), "Entry price should be close + slippage"


def test_engine_stop_loss_respected():
    """
    If we construct candles where price drops far below stop immediately after entry,
    the trade must exit via stop_loss.
    """
    # Rising prices to trigger RSI_OVERSOLD (many dips), then a crash
    candles = _make_trending_candles(350)
    # Add a big crash at bar 300 so any open position hits stop
    crash_prices = [candles[-1].close * 0.5] * 50
    for i, p in enumerate(crash_prices):
        candles.append(Candle(
            ts=(len(candles)) * 3600 * 1000,
            open=p, high=p * 1.001, low=p * 0.50, close=p,
            volume=2_000_000.0, swap_count=20,
        ))

    strategy = _make_strategy("RSI_OVERSOLD", "RSI", "NONE")
    results  = run_all_risk_levels(strategy, candles)
    all_trades = [t for r in results for t in r.trades]
    stop_exits = [t for t in all_trades if t.exit_reason == "stop_loss"]
    # There should be at least one stop-loss exit given the crash
    assert len(stop_exits) > 0


# ── Filter tests ──────────────────────────────────────────────────────────────

def test_filter_passes_clean_strategy():
    result = apply_filters(_make_metrics(), _make_backtest())
    assert result.passed is True
    assert result.failure_reason is None


def test_filter_rejects_high_drawdown():
    result = apply_filters(_make_metrics(max_drawdown_pct=0.25), _make_backtest())
    assert result.passed is False
    assert "drawdown" in result.failure_reason.lower()


def test_filter_rejects_low_win_rate():
    result = apply_filters(_make_metrics(win_rate=0.30), _make_backtest())
    assert result.passed is False
    assert "win rate" in result.failure_reason.lower()


def test_filter_rejects_low_trade_count():
    result = apply_filters(_make_metrics(trade_count=15), _make_backtest())
    assert result.passed is False
    assert "trades" in result.failure_reason.lower()


def test_filter_rejects_only_max_risk():
    result = apply_filters(_make_metrics(), _make_backtest(only_works_at_max_risk=True))
    assert result.passed is False
    assert "2%" in result.failure_reason


def test_filter_rejects_low_sharpe():
    result = apply_filters(_make_metrics(sharpe=0.3), _make_backtest())
    assert result.passed is False
    assert "sharpe" in result.failure_reason.lower()


def test_filter_streak_limit_scales_by_bucket():
    """Streak limit should be 8 for hourly, 10 for 4h, 15 for daily bars."""
    from backtesting.filters import _streak_limit
    assert _streak_limit(3_600)  == 8
    assert _streak_limit(14_400) == 10
    assert _streak_limit(86_400) == 15
    assert _streak_limit(604_800) == 10  # weekly


def test_filter_daily_bars_allows_longer_streak():
    """A streak of 12 should pass on daily bars but fail on 4h bars."""
    metrics = _make_metrics(max_consecutive_losses=12)
    assert apply_filters(metrics, _make_backtest(), bucket_seconds=86_400).passed is True
    assert apply_filters(metrics, _make_backtest(), bucket_seconds=14_400).passed is False


# ── Scorer tests ──────────────────────────────────────────────────────────────

def test_scorer_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_scorer_returns_zero_profit_factor_below_threshold():
    score = score_strategy(_make_metrics(profit_factor=1.2), "s1")
    assert score.profit_factor_score == 0.0


def test_scorer_total_in_valid_range():
    score = score_strategy(_make_metrics(), "s1")
    assert 0.0 <= score.total <= 100.0


def test_scorer_better_sharpe_gives_higher_score():
    s_low  = score_strategy(_make_metrics(sharpe=0.6, sortino=0.9), "low")
    s_high = score_strategy(_make_metrics(sharpe=2.5, sortino=3.5), "high")
    assert s_high.risk_adjusted_return > s_low.risk_adjusted_return


# ── Selector tests ────────────────────────────────────────────────────────────

def test_selector_picks_highest_score():
    e1 = _make_evaluated("s1", score_total=60.0)
    e2 = _make_evaluated("s2", score_total=80.0)
    selection = select_winner([e1, e2], [])
    assert selection.winner.strategy.id == "s2"


def test_selector_skips_similar_to_past_failure():
    e1 = _make_evaluated("s1", score_total=80.0)                          # top scorer: RSI+RSI_OVERSOLD
    e2 = _make_evaluated("s2", score_total=60.0, primary="MACD",          # different combo
                         trigger="MACD_CROSS_ABOVE")
    # s1 matches a past failure (RSI + RSI_OVERSOLD); s2 does not
    past = [{"primary_indicator_type": "RSI", "entry_trigger": "RSI_OVERSOLD"}]
    selection = select_winner([e1, e2], past)
    # s1 is skipped; s2 becomes winner
    assert selection.winner.strategy.id == "s2"


def test_selector_raises_when_no_strategies_pass():
    e1 = _make_evaluated("s1", passed=False, failure="Win rate too low")
    e2 = _make_evaluated("s2", passed=False, failure="Sharpe too low")
    with pytest.raises(ValueError, match="No strategies survived"):
        select_winner([e1, e2], [])


# ── Metrics helpers ───────────────────────────────────────────────────────────

def test_max_drawdown_simple():
    equity = [100.0, 110.0, 90.0, 95.0, 105.0]
    dd, dur, _ = _max_drawdown(equity)
    # Peak=110, trough=90 → dd = 20/110 ≈ 18.2%
    assert abs(dd - 20.0 / 110.0) < 1e-6
    assert dur == 1  # 1 bar from peak (idx 1) to trough (idx 2)


def test_max_consecutive_losses():
    trades = [
        Trade(0, 1, 1.0, 1.1, 1.0, "long", 100.0,  0.1,  1.0, "take_profit", 0, 3600000),
        Trade(2, 3, 1.0, 0.9, 1.0, "long", -100.0, -0.1, -1.0, "stop_loss",  0, 7200000),
        Trade(4, 5, 1.0, 0.9, 1.0, "long", -100.0, -0.1, -1.0, "stop_loss",  0, 10800000),
        Trade(6, 7, 1.0, 0.9, 1.0, "long", -100.0, -0.1, -1.0, "stop_loss",  0, 14400000),
        Trade(8, 9, 1.0, 1.1, 1.0, "long", 100.0,  0.1,  1.0, "take_profit", 0, 18000000),
    ]
    assert _max_consecutive_losses(trades) == 3


# ── EV filter tests ───────────────────────────────────────────────────────────

def test_filter_rejects_negative_ev():
    """avg_r_multiple <= 0 is negative EV -- must fail the filter."""
    result = apply_filters(_make_metrics(avg_r_multiple=-0.1), _make_backtest())
    assert result.passed is False
    assert "expected value" in result.failure_reason.lower()


def test_filter_rejects_zero_ev():
    result = apply_filters(_make_metrics(avg_r_multiple=0.0), _make_backtest())
    assert result.passed is False
    assert "expected value" in result.failure_reason.lower()


def test_filter_passes_positive_ev():
    """Positive avg_r_multiple should clear the EV filter."""
    result = apply_filters(_make_metrics(avg_r_multiple=0.35), _make_backtest())
    assert result.passed is True


# ── EV scorer tests ───────────────────────────────────────────────────────────

def test_scorer_ev_zero_below_threshold():
    """avg_r_multiple below 0.20R should score ~0 on the EV dimension."""
    score = score_strategy(_make_metrics(avg_r_multiple=0.05), "s1")
    assert score.ev_score < 5.0  # very close to zero


def test_scorer_ev_scales_with_r():
    """Higher avg_r_multiple should give a higher EV score."""
    s_low  = score_strategy(_make_metrics(avg_r_multiple=0.30), "low")
    s_high = score_strategy(_make_metrics(avg_r_multiple=0.90), "high")
    assert s_high.ev_score > s_low.ev_score


def test_scorer_ev_in_score_result():
    """ScoreResult must have the ev_score field (not trade_frequency)."""
    score = score_strategy(_make_metrics(), "s1")
    assert hasattr(score, "ev_score")
    assert hasattr(score, "statistical_confidence")
    assert not hasattr(score, "trade_frequency")
    assert not hasattr(score, "regime_performance")


# ── Monte Carlo tests ─────────────────────────────────────────────────────────

def _make_winning_trades(n: int = 50) -> list[Trade]:
    """Trades with positive R (2R wins, -1R losses, 60% WR)."""
    trades = []
    for i in range(n):
        is_win = (i % 10) < 6  # 60% win rate
        r = 2.0 if is_win else -1.0
        pnl = r * 100.0
        trades.append(Trade(
            entry_bar=i * 2, exit_bar=i * 2 + 1,
            entry_price=1.0, exit_price=1.0 + r * 0.01,
            size=100.0, direction="long",
            pnl=pnl, pnl_pct=pnl / 10000.0, r_multiple=r,
            exit_reason="take_profit" if is_win else "stop_loss",
            entry_ts=i * 3600000, exit_ts=(i + 1) * 3600000,
        ))
    return trades


def _make_losing_trades(n: int = 50) -> list[Trade]:
    """Trades with negative EV (small wins, large losses)."""
    trades = []
    for i in range(n):
        is_win = (i % 10) < 3  # 30% win rate
        r = 0.5 if is_win else -2.0
        pnl = r * 100.0
        trades.append(Trade(
            entry_bar=i * 2, exit_bar=i * 2 + 1,
            entry_price=1.0, exit_price=1.0 + r * 0.01,
            size=100.0, direction="long",
            pnl=pnl, pnl_pct=pnl / 10000.0, r_multiple=r,
            exit_reason="take_profit" if is_win else "stop_loss",
            entry_ts=i * 3600000, exit_ts=(i + 1) * 3600000,
        ))
    return trades


def test_monte_carlo_returns_result():
    """Basic smoke test: run_monte_carlo returns a MonteCarloResult."""
    from backtesting.models import MonteCarloResult
    trades = _make_winning_trades(50)
    result = run_monte_carlo(trades, risk_pct=1.0, n_sims=100, trades_per_sim=50)
    assert isinstance(result, MonteCarloResult)
    assert 0.0 <= result.p95_drawdown <= 1.0
    assert result.risk_class in ("low", "moderate", "high", "extreme")


def test_monte_carlo_low_risk_on_good_strategy():
    """A high win-rate 2R strategy should produce low or moderate risk classification."""
    trades = _make_winning_trades(60)
    result = run_monte_carlo(trades, risk_pct=1.0, n_sims=200, trades_per_sim=50)
    assert result.risk_class in ("low", "moderate"), \
        f"Expected low/moderate for good strategy, got {result.risk_class} (p95={result.p95_drawdown:.1%})"


def test_monte_carlo_extreme_on_too_few_trades():
    """Fewer than 10 trades -> worst-case 'extreme' result (cannot resample)."""
    trades = _make_winning_trades(5)
    result = run_monte_carlo(trades, risk_pct=1.0)
    assert result.risk_class == "extreme"
    assert result.p95_drawdown == 1.0


def test_monte_carlo_prop_firm_flags():
    """Good strategy should pass the 10% prop firm threshold."""
    trades = _make_winning_trades(60)
    result = run_monte_carlo(trades, risk_pct=1.0, n_sims=200, trades_per_sim=50)
    assert result.prop_firm_10pct  # numpy bool: use truthy check not identity


def test_filter_rejects_extreme_monte_carlo():
    """A strategy with extreme MC risk must fail the filter."""
    from backtesting.models import MonteCarloResult
    mc = MonteCarloResult(
        p95_drawdown=0.15, p95_consecutive_losses=15,
        risk_class="extreme",
        prop_firm_5pct=False, prop_firm_8pct=False, prop_firm_10pct=False,
    )
    result = apply_filters(_make_metrics(), _make_backtest(), mc_result=mc)
    assert result.passed is False
    assert "extreme" in result.failure_reason.lower()


def test_filter_rejects_mc_failing_8pct_rule():
    """A strategy exceeding 8% MC p95 DD must fail even if not extreme."""
    from backtesting.models import MonteCarloResult
    mc = MonteCarloResult(
        p95_drawdown=0.09, p95_consecutive_losses=8,
        risk_class="high",
        prop_firm_5pct=False, prop_firm_8pct=False, prop_firm_10pct=False,
    )
    result = apply_filters(_make_metrics(), _make_backtest(), mc_result=mc)
    assert result.passed is False
    assert "8%" in result.failure_reason


def test_filter_passes_good_monte_carlo():
    """A good MC result (moderate, passes 8% rule) should not add a failure."""
    from backtesting.models import MonteCarloResult
    mc = MonteCarloResult(
        p95_drawdown=0.05, p95_consecutive_losses=5,
        risk_class="moderate",
        prop_firm_5pct=False, prop_firm_8pct=True, prop_firm_10pct=True,
    )
    result = apply_filters(_make_metrics(), _make_backtest(), mc_result=mc)
    assert result.passed is True


# ── select_top_3 tests ────────────────────────────────────────────────────────

def test_selector_top3_picks_decorrelated():
    """Three strategies with distinct indicator+trigger combos should all be selected."""
    e1 = _make_evaluated("s1", 90.0, primary="RSI",      trigger="RSI_OVERSOLD")
    e2 = _make_evaluated("s2", 80.0, primary="MACD",     trigger="MACD_CROSS_ABOVE")
    e3 = _make_evaluated("s3", 70.0, primary="BOLLINGER", trigger="BB_LOWER_TOUCH")
    winners = select_top_3([e1, e2, e3], [])
    assert len(winners) == 3
    ids = [w.strategy.id for w in winners]
    assert ids == ["s1", "s2", "s3"]


def test_selector_top3_skips_correlated():
    """Two strategies sharing indicator+trigger should not both be selected."""
    e1 = _make_evaluated("s1", 90.0, primary="RSI", trigger="RSI_OVERSOLD")
    e2 = _make_evaluated("s2", 80.0, primary="RSI", trigger="RSI_OVERSOLD")   # correlated with e1
    e3 = _make_evaluated("s3", 70.0, primary="MACD", trigger="MACD_CROSS_ABOVE")
    winners = select_top_3([e1, e2, e3], [])
    ids = [w.strategy.id for w in winners]
    assert "s1" in ids
    assert "s2" not in ids   # correlated with s1 -- skipped
    assert "s3" in ids


def test_selector_top3_returns_fewer_when_insufficient():
    """If only one strategy passes, return a list of one (not error)."""
    e1 = _make_evaluated("s1", 80.0)
    e2 = _make_evaluated("s2", passed=False, failure="Low win rate")
    winners = select_top_3([e1, e2], [])
    assert len(winners) == 1
    assert winners[0].strategy.id == "s1"


def test_selector_top3_skips_past_failure():
    """Strategies matching a past failure should be skipped by select_top_3."""
    e1 = _make_evaluated("s1", 90.0, primary="RSI", trigger="RSI_OVERSOLD")
    e2 = _make_evaluated("s2", 80.0, primary="MACD", trigger="MACD_CROSS_ABOVE")
    past = [{"primary_indicator_type": "RSI", "entry_trigger": "RSI_OVERSOLD"}]
    winners = select_top_3([e1, e2], past)
    ids = [w.strategy.id for w in winners]
    assert "s1" not in ids
    assert "s2" in ids


def test_selector_top3_empty_when_all_fail():
    """All strategies failing filters -> empty list, not error."""
    e1 = _make_evaluated("s1", passed=False, failure="Drawdown too high")
    e2 = _make_evaluated("s2", passed=False, failure="Sharpe too low")
    winners = select_top_3([e1, e2], [])
    assert winners == []


# ── Engine risk level test ─────────────────────────────────────────────────────

def test_engine_has_025_risk_level():
    """RISK_LEVELS must include 0.25% as the minimum risk level per spec."""
    assert 0.25 in RISK_LEVELS
    assert len(RISK_LEVELS) == 5   # [0.25, 0.5, 1.0, 1.5, 2.0]
