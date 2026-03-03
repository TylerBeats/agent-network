"""
Per-asset class profiles for the Strategy Builder.

Maps each chain to an asset class, then provides class-specific hints that
guide the LLM toward indicator types and triggers that historically work for
that market -- and explicitly warns it off patterns known to fail.
"""
from __future__ import annotations

# -- Asset class map -----------------------------------------------------------

ASSET_CLASS: dict[str, str] = {
    # Commodities -- strong secular trends, wide ATR, mean-reversion fails badly
    "gold":         "commodity",

    # Broad equity indices -- long uptrend + periodic bear markets
    "spx500":       "equity_etf",
    "spx500_full":  "equity_index",   # ^GSPC -- no real volume
    "nasdaq":       "equity_etf",

    # Individual stocks / ETFs -- user-supplied ticker
    "stocks":       "equity_stock",

    # Crypto on-chain
    "pulsechain":   "crypto",
    "ethereum":     "crypto",
    "arbitrum":     "crypto",
    "base":         "crypto",
    "polygon":      "crypto",
    "bsc":          "crypto",
    "avalanche":    "crypto",
    "solana":       "crypto",
    "crypto_yahoo": "crypto",
}

# -- Per-class strategy hints --------------------------------------------------

_HINTS: dict[str, dict] = {
    "commodity": {
        "class_label": "Commodity (Gold Futures)",
        "regime_note": (
            "Gold is a strongly trending asset over multi-year periods ($280->$2900 from 2000–2026). "
            "It spends long stretches in sustained uptrends with periodic sharp corrections. "
            "Mean-reversion strategies consistently fail: the asset trends far beyond normal "
            "oscillator thresholds before reversing."
        ),
        "preferred_indicators": ["EMA", "SMA", "MACD", "ADX", "ATR", "DONCHIAN"],
        "preferred_triggers":   ["PRICE_ABOVE_EMA", "MACD_CROSS_ABOVE", "DC_UPPER_BREAK", "MACD_CROSS_BELOW", "DC_LOWER_BREAK"],
        "preferred_confirmations": ["ADX_TRENDING", "ATR_EXPANDING"],
        "avoid_triggers": [
            "RSI_OVERSOLD", "RSI_OVERBOUGHT",
            "BB_LOWER_TOUCH", "BB_UPPER_TOUCH",
            "CCI_OVERSOLD", "CCI_OVERBOUGHT",
            "STOCH_OVERSOLD", "STOCH_OVERBOUGHT",
            "WILLR_OVERSOLD", "WILLR_OVERBOUGHT",
            "KB_LOWER_TOUCH", "KB_UPPER_TOUCH",
        ],
        "avoid_reason": (
            "Mean-reversion oscillator triggers (RSI oversold/overbought, Bollinger band touch, "
            "CCI, Stochastic, Williams %R, Keltner touch) consistently fail on gold because the "
            "asset can remain 'overbought' for months during trends. Avoid all of these."
        ),
        "strategy_focus": (
            "Focus on TREND-FOLLOWING: EMA crossovers, MACD crossovers, Donchian channel breakouts. "
            "Use ADX_TRENDING as a confirmation filter to only trade when a trend is established. "
            "Use wide ATR multiples for stop losses (2–3xATR) to avoid being stopped out by noise. "
            "Trailing stops work well for capturing extended moves. "
            "Both long and short sides are valid -- gold has major bear phases too (2011–2015)."
        ),
        "volume_note": "Volume is in futures contracts -- VOLUME_ABOVE_SMA is a valid signal.",
    },

    "equity_etf": {
        "class_label": "Equity ETF (S&P 500 / Nasdaq)",
        "regime_note": (
            "Broad equity ETFs (SPY, QQQ) have a strong long-term uptrend but with periodic "
            "bear markets of 20–50% (2000–2002, 2008–2009, 2020, 2022). "
            "Both trend-following and mild mean-reversion strategies can work."
        ),
        "preferred_indicators": ["EMA", "MACD", "RSI", "SMA", "ATR", "BOLLINGER"],
        "preferred_triggers":   [
            "PRICE_ABOVE_EMA", "MACD_CROSS_ABOVE", "MACD_CROSS_BELOW",
            "RSI_OVERSOLD", "DC_UPPER_BREAK",
        ],
        "preferred_confirmations": ["ADX_TRENDING", "VOLUME_ABOVE_SMA", "ATR_EXPANDING"],
        "avoid_triggers": [],
        "avoid_reason": "",
        "strategy_focus": (
            "Mix trend-following (EMA crossovers, MACD) with mild mean-reversion (RSI oversold "
            "during bull markets). Bias toward LONG-only or asymmetric long-heavy strategies -- "
            "equity ETFs have a structural upward drift. Use RSI_OVERSOLD carefully: only effective "
            "as a buying dip strategy in uptrends, not as a standalone reversal signal."
        ),
        "volume_note": "Volume is in shares -- VOLUME_ABOVE_SMA is a valid and useful signal.",
    },

    "equity_index": {
        "class_label": "Equity Index (^GSPC -- full history ~1927)",
        "regime_note": (
            "The S&P 500 index has ~100 years of history spanning the Great Depression, WWII, "
            "1970s stagflation, dot-com crash, 2008 financial crisis, and COVID. "
            "The full dataset spans wildly different market regimes -- strategies must be robust."
        ),
        "preferred_indicators": ["EMA", "SMA", "MACD", "RSI", "ATR"],
        "preferred_triggers":   ["PRICE_ABOVE_EMA", "MACD_CROSS_ABOVE", "RSI_OVERSOLD", "DC_UPPER_BREAK"],
        "preferred_confirmations": ["ADX_TRENDING", "ATR_EXPANDING", "NONE"],
        "avoid_triggers": [],
        "avoid_reason": "",
        "strategy_focus": (
            "With 100 years of data the strategy must survive multiple regime shifts. "
            "Prefer simple, robust trend-following systems (EMA crossovers, MACD) with "
            "wide stops that can weather large drawdowns. "
            "IMPORTANT: Volume data for ^GSPC is synthetic -- do NOT use VOLUME_ABOVE_SMA."
        ),
        "volume_note": "CRITICAL: Volume is synthetic for ^GSPC -- do NOT use VOLUME_ABOVE_SMA as a confirmation.",
    },

    "equity_stock": {
        "class_label": "Individual US Equity / ETF",
        "regime_note": (
            "Individual stocks can trend strongly (growth stocks) or trade sideways for years "
            "(value stocks, cyclicals). Volatility and behaviour vary widely by sector."
        ),
        "preferred_indicators": ["EMA", "MACD", "RSI", "SMA", "BOLLINGER", "ATR"],
        "preferred_triggers":   [
            "PRICE_ABOVE_EMA", "MACD_CROSS_ABOVE", "RSI_OVERSOLD",
            "BB_LOWER_TOUCH", "DC_UPPER_BREAK",
        ],
        "preferred_confirmations": ["VOLUME_ABOVE_SMA", "ADX_TRENDING", "ATR_EXPANDING"],
        "avoid_triggers": [],
        "avoid_reason": "",
        "strategy_focus": (
            "Generate a mix of trend-following and mean-reversion strategies. "
            "VOLUME_ABOVE_SMA is highly valuable for individual stocks -- volume confirmation "
            "helps filter false breakouts and low-conviction moves."
        ),
        "volume_note": "Volume is in shares -- VOLUME_ABOVE_SMA is a strong and reliable signal for individual stocks.",
    },

    "crypto": {
        "class_label": "Cryptocurrency",
        "regime_note": (
            "Crypto markets are highly volatile with explosive trends (10x+ moves), deep bear "
            "markets (70–90% drawdowns), and extended sideways chop. All strategy types can "
            "work depending on the regime. Consecutive losing streaks can be long during "
            "choppy markets, so streak-resistant strategies are preferred."
        ),
        "preferred_indicators": ["RSI", "EMA", "MACD", "BOLLINGER", "ATR", "STOCH", "OBV"],
        "preferred_triggers":   [
            "RSI_OVERSOLD", "PRICE_ABOVE_EMA", "MACD_CROSS_ABOVE",
            "BB_LOWER_TOUCH", "DC_UPPER_BREAK", "ROC_CROSS_ABOVE",
        ],
        "preferred_confirmations": ["VOLUME_ABOVE_SMA", "OBV_RISING", "ADX_TRENDING", "ATR_EXPANDING"],
        "avoid_triggers": [],
        "avoid_reason": "",
        "strategy_focus": (
            "All indicator types are viable. Prioritise strategies with strict confirmation "
            "filters to reduce false signals during choppy regimes. "
            "OBV_RISING is particularly useful for on-chain assets where volume reflects "
            "genuine DEX activity. Use ATR-based stops (2–3x) to survive high volatility."
        ),
        "volume_note": "Volume from on-chain subgraphs reflects real DEX swap activity in USD -- VOLUME_ABOVE_SMA and OBV_RISING are reliable signals.",
    },
}

# Fallback for unknown chains
_DEFAULT_HINTS = _HINTS["crypto"]


def get_asset_class(chain: str) -> str:
    """Return the asset class string for a given chain name."""
    return ASSET_CLASS.get(chain, "crypto")


def get_asset_hints(chain: str) -> dict:
    """Return the strategy hint dict for a given chain name."""
    cls = get_asset_class(chain)
    return _HINTS.get(cls, _DEFAULT_HINTS)


def build_asset_key(chain: str, token_address: str) -> str:
    """
    Build a short, filesystem-safe key uniquely identifying this asset.

    Examples:
      gold           -> "gold"
      spx500         -> "spx500"
      pulsechain + 0x2b591e... -> "pulsechain_0x2b591e99"
      stocks + AAPL  -> "stocks_AAPL"
    """
    if token_address:
        # Shorten long hex addresses; keep user-supplied tickers as-is
        addr = token_address[:10] if token_address.startswith("0x") else token_address
        return f"{chain}_{addr}".replace("/", "-")
    return chain


def summarise_strategy_log(log: list[dict]) -> str:
    """
    Produce a compact textual summary of per-asset strategy history for the prompt.

    Groups by primary_indicator and entry_trigger to show pass/fail counts and
    best metrics, giving the LLM actionable feedback without flooding the context.
    """
    if not log:
        return ""

    from collections import defaultdict

    ind_stats:  dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0, "scores": [], "failures": []})
    trig_stats: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0})

    for entry in log:
        ind  = entry.get("primary_indicator", "?")
        trig = entry.get("entry_trigger", "?")
        passed = entry.get("result") == "passed"

        if passed:
            ind_stats[ind]["pass"]  += 1
            trig_stats[trig]["pass"] += 1
            score = entry.get("score", 0)
            if score:
                ind_stats[ind]["scores"].append(score)
        else:
            ind_stats[ind]["fail"]  += 1
            trig_stats[trig]["fail"] += 1
            reason = entry.get("failure_reason", "")
            if reason and reason not in ind_stats[ind]["failures"]:
                ind_stats[ind]["failures"].append(reason[:60])

    lines = [f"ASSET STRATEGY LOG ({len(log)} strategies evaluated across all cycles):"]

    # Indicator summary
    lines.append("\n  Indicator pass/fail rates:")
    for ind, st in sorted(ind_stats.items(), key=lambda x: -(x[1]["pass"])):
        total = st["pass"] + st["fail"]
        avg_score = f"  avg score {sum(st['scores'])/len(st['scores']):.0f}" if st["scores"] else ""
        fail_note = f"  failure: {st['failures'][0]}" if st["failures"] else ""
        lines.append(f"    {ind:<12} {st['pass']}/{total} passed{avg_score}{fail_note}")

    # Trigger summary
    lines.append("\n  Trigger pass/fail rates:")
    for trig, st in sorted(trig_stats.items(), key=lambda x: -(x[1]["pass"])):
        total = st["pass"] + st["fail"]
        lines.append(f"    {trig:<30} {st['pass']}/{total} passed")

    lines.append(
        "\nUse this data to BIAS your generation toward indicators and triggers with "
        "higher pass rates, and AVOID those with 0 passes and multiple failures."
    )

    return "\n".join(lines)
