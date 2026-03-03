from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Strategy:
    id: str
    name: str
    primary_indicator: dict       # {"type": "RSI", "params": {"period": 14}}
    confirmation_indicator: dict  # {"type": "EMA", "params": {"period": 50}}
    entry: dict                   # {"trigger": "RSI_OVERSOLD", "filter": "PRICE_ABOVE_EMA"}
    exit: dict                    # {"stop_loss": {...}, "take_profit": {...},
                                  #  "trailing_stop_atr": null, "time_exit_bars": 48}
    risk: dict                    # {"max_open_positions": 3}
    metadata: dict = field(default_factory=dict)


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    size: float           # token units
    direction: str        # "long"
    pnl: float            # USD P&L after commission
    pnl_pct: float        # fraction of equity at entry
    r_multiple: float     # pnl / initial_risk_usd
    exit_reason: str      # "stop_loss" | "take_profit" | "trailing_stop" | "time_exit" | "end_of_data"
    entry_ts: int         # Unix ms
    exit_ts: int
    regime: str = "unknown"  # "bull" | "bear" | "sideways"


@dataclass
class RiskLevelResult:
    risk_pct: float
    trades: list[Trade]
    equity_curve: list[float]
    sharpe: float


@dataclass
class BacktestResult:
    strategy_id: str
    best_risk_pct: float         # selected from [0.5, 1.0, 1.5, 2.0]
    trades: list[Trade]
    equity_curve: list[float]    # equity value after each bar (IS series)
    in_sample_bars: int
    out_of_sample_bars: int
    only_works_at_max_risk: bool  # True if 2.0% is the only risk level that passes hard filters
    oos_sharpe: float = 0.0       # Sharpe ratio on the held-out 20% OOS period
    oos_trade_count: int = 0      # Number of trades in OOS period
    oos_win_rate: float = 0.0     # Win rate in OOS period
    confidence_rating: str = "Medium"  # "High" | "Medium" | "Low"


@dataclass
class MetricsResult:
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    max_drawdown_recovery_bars: int
    win_rate: float
    profit_factor: float
    avg_r_multiple: float
    trade_count: int
    max_consecutive_losses: int
    worst_trade_pct: float        # worst single-trade loss as fraction of equity
    monthly_return_variance: float
    bull_pnl_pct: float           # total P&L in bull-regime trades / initial capital
    bear_pnl_pct: float
    sideways_pnl_pct: float
    best_risk_pct: float
    total_pnl_pct: float = 0.0   # total return over full backtest: (final - initial) / initial
    total_pnl_usd: float = 0.0   # total USD P&L over full backtest


@dataclass
class FilterResult:
    passed: bool
    failure_reason: str | None = None  # None when passed


@dataclass
class MonteCarloResult:
    """
    Results from 800-simulation Monte Carlo drawdown analysis.
    All drawdown figures are fractions (e.g. 0.07 = 7%).
    """
    p95_drawdown: float          # 95th percentile max drawdown across all simulations
    p95_consecutive_losses: int  # 95th percentile max losing streak
    risk_class: str              # "low" | "moderate" | "high" | "extreme"
    prop_firm_5pct: bool         # True if p95 DD < 5%
    prop_firm_8pct: bool         # True if p95 DD < 8%
    prop_firm_10pct: bool        # True if p95 DD < 10%
    n_sims: int = 800
    n_trades: int = 100          # trades resampled per simulation


@dataclass
class ScoreResult:
    strategy_id: str
    total: float                    # 0-100 weighted composite
    ev_score: float                 # 25% -- expected value per trade (avg R multiple based)
    drawdown_profile: float         # 25% -- MC 95th pct DD if available, else historical
    risk_adjusted_return: float     # 20% -- Sharpe + Sortino
    consistency: float              # 15% -- monthly return variance + regime stability
    profit_factor_score: float      # 10% -- gross profit / gross loss
    statistical_confidence: float   # 5%  -- trade count relative to minimum sample


@dataclass
class EvaluatedStrategy:
    strategy: Strategy
    backtest: BacktestResult
    metrics: MetricsResult
    filter_result: FilterResult
    score: ScoreResult | None     # None when filtered out
    mc_result: MonteCarloResult | None = None


@dataclass
class StrategySelection:
    winners: list[EvaluatedStrategy]   # top 1-3 survivors, ordered by score descending
    all_evaluated: list[EvaluatedStrategy]
    selection_note: str

    # Backward-compatible properties
    @property
    def winner(self) -> EvaluatedStrategy:
        return self.winners[0]

    @property
    def runner_up(self) -> EvaluatedStrategy | None:
        return self.winners[1] if len(self.winners) > 1 else None
