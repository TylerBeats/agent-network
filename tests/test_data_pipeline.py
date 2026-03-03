"""
Unit tests for the Phase 2 data pipeline.
All HTTP calls are mocked — no real network requests are made.
"""
from unittest.mock import MagicMock, patch

import pytest

from data.candles import Candle, build_candles_from_swaps
from data.quality import DataQualityResult, analyse_data_quality
from data.pipeline import fetch_candles


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_post(json_body: dict) -> MagicMock:
    """Return a mock requests.post that returns the given JSON."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = json_body
    mock = MagicMock(return_value=mock_resp)
    return mock


def _mock_get(json_body: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = json_body
    mock = MagicMock(return_value=mock_resp)
    return mock


def _make_candles(specs: list[tuple]) -> list[Candle]:
    """Build candles from (ts, open, high, low, close, volume, swap_count) tuples."""
    return [Candle(ts=s[0], open=s[1], high=s[2], low=s[3], close=s[4], volume=s[5], swap_count=s[6]) for s in specs]


# ── VWAP candle construction ──────────────────────────────────────────────────

def test_vwap_calculation_v2():
    """
    Three PulseX v2 swaps in one bucket.
    raw_price = (amountUSD / 2) / tokenVol
    VWAP close = Σ(price × usd) / Σ(usd)
    """
    # Each swap: 1 token, prices 1.0 / 2.0 / 3.0 USD (amountUSD = 2× because v2 counts both sides)
    swaps = [
        {"timestamp": "1000", "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "2.0"},
        {"timestamp": "1001", "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "4.0"},
        {"timestamp": "1002", "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "6.0"},
    ]
    candles = build_candles_from_swaps(swaps, bucket_seconds=3600, is_token0=True, is_v2=True)
    assert len(candles) == 1
    c = candles[0]
    # raw prices: 1.0, 2.0, 3.0  |  weights: 2.0, 4.0, 6.0
    # VWAP = (1.0×2.0 + 2.0×4.0 + 3.0×6.0) / (2.0 + 4.0 + 6.0) = 28/12 ≈ 2.333
    assert abs(c.close - (28 / 12)) < 1e-9
    assert c.open == 1.0
    assert c.high == 3.0
    assert c.low == 1.0
    assert c.swap_count == 3


def test_vwap_calculation_v3():
    """
    Uniswap v3 swaps use signed amount0/amount1 and amountUSD is one side only.
    raw_price = amountUSD / abs(amount0)
    """
    swaps = [
        {"timestamp": "1000", "amount0": "-2", "amount1": "200", "amountUSD": "200"},  # price 100
        {"timestamp": "1001", "amount0": "-1", "amount1": "150", "amountUSD": "150"},  # price 150
    ]
    candles = build_candles_from_swaps(swaps, bucket_seconds=3600, is_token0=True, is_v2=False)
    assert len(candles) == 1
    c = candles[0]
    # VWAP = (100×200 + 150×150) / (200 + 150) = (20000 + 22500) / 350 = 42500/350 ≈ 121.43
    assert abs(c.close - (42500 / 350)) < 1e-9


def test_swaps_split_into_separate_buckets():
    """Swaps in different 1-hour buckets must produce separate candles."""
    swaps = [
        {"timestamp": "0",    "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "2.0"},
        {"timestamp": "3600", "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "4.0"},
    ]
    candles = build_candles_from_swaps(swaps, bucket_seconds=3600, is_token0=True, is_v2=True)
    assert len(candles) == 2


def test_zero_amount_usd_swaps_skipped():
    swaps = [
        {"timestamp": "1000", "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "0"},
        {"timestamp": "1001", "amount0In": "1", "amount0Out": "0", "amount1In": "0", "amount1Out": "0", "amountUSD": "2.0"},
    ]
    candles = build_candles_from_swaps(swaps, bucket_seconds=3600, is_token0=True, is_v2=True)
    assert len(candles) == 1
    assert candles[0].swap_count == 1


# ── Pair ranking ──────────────────────────────────────────────────────────────

def test_pair_ranking_prefers_stablecoin():
    """PulseX pair detection should pick the stablecoin partner over WPLS."""
    from data.sources.pulsex import _pair_rank
    from data.chains import PULSEX_STABLECOINS, WPLS

    stable_addr = next(iter(PULSEX_STABLECOINS))
    stablecoin_pair = {
        "id": "0xaaa",
        "volumeUSD": "1000",
        "token0": {"id": "0xtoken", "symbol": "TKN"},
        "token1": {"id": stable_addr, "symbol": "DAI"},
        "is_token0": True,
    }
    wpls_pair = {
        "id": "0xbbb",
        "volumeUSD": "9999",
        "token0": {"id": "0xtoken", "symbol": "TKN"},
        "token1": {"id": WPLS, "symbol": "WPLS"},
        "is_token0": True,
    }
    # Stablecoin pair should rank higher even with lower volume
    assert _pair_rank(stablecoin_pair) > _pair_rank(wpls_pair)


def test_pair_ranking_falls_back_to_wpls():
    """When no stablecoin pair exists, WPLS pair should rank above exotic."""
    from data.sources.pulsex import _pair_rank
    from data.chains import WPLS

    wpls_pair = {
        "id": "0xbbb",
        "volumeUSD": "1000",
        "token0": {"id": "0xtoken", "symbol": "TKN"},
        "token1": {"id": WPLS, "symbol": "WPLS"},
        "is_token0": True,
    }
    exotic_pair = {
        "id": "0xccc",
        "volumeUSD": "9999",
        "token0": {"id": "0xtoken", "symbol": "TKN"},
        "token1": {"id": "0xsomeother", "symbol": "OTHER"},
        "is_token0": True,
    }
    assert _pair_rank(wpls_pair) > _pair_rank(exotic_pair)


def test_pool_ranking_prefers_stablecoin():
    """Uniswap v3 pool detection should prefer stablecoin partners."""
    from data.sources.thegraph import _pool_rank
    from data.chains import CHAIN_STABLES, CHAIN_WNATIVE

    stables = CHAIN_STABLES["ethereum"]
    wnative = CHAIN_WNATIVE["ethereum"]
    stable_addr = next(iter(stables))

    stable_pool = {
        "id": "0xpool1",
        "totalValueLockedUSD": "1000",
        "feeTier": "3000",
        "token0": {"id": "0xtoken", "symbol": "TKN"},
        "token1": {"id": stable_addr, "symbol": "USDC"},
        "is_token0": True,
    }
    native_pool = {
        "id": "0xpool2",
        "totalValueLockedUSD": "9999",
        "feeTier": "3000",
        "token0": {"id": "0xtoken", "symbol": "TKN"},
        "token1": {"id": wnative, "symbol": "WETH"},
        "is_token0": True,
    }
    assert _pool_rank(stable_pool, stables, wnative) > _pool_rank(native_pool, stables, wnative)


# ── Data quality checks ───────────────────────────────────────────────────────

def test_quality_passes_clean_data():
    candles = _make_candles([
        (0,     1.0, 1.1, 0.9, 1.05, 1000, 5),
        (3600000, 1.05, 1.15, 0.95, 1.10, 1200, 6),
    ])
    result = analyse_data_quality(candles, "subgraph")
    assert result.passed is True
    assert result.confidence == "high"
    assert result.warnings == []


def test_quality_spike_detected():
    """A candle with high/low >= 5 should trigger a warning."""
    candles = _make_candles([
        (0,       1.0, 6.0, 1.0, 1.5, 1000, 5),   # high/low = 6× — spike
        (3600000, 1.5, 1.6, 1.4, 1.55, 800, 4),
    ])
    result = analyse_data_quality(candles, "subgraph")
    assert result.passed is False
    assert any("high/low" in w for w in result.warnings)


def test_quality_jump_detected():
    """A 10× close-to-close jump between candles should trigger a warning."""
    candles = _make_candles([
        (0,       1.0, 1.1, 0.9, 1.0,  1000, 5),
        (3600000, 1.0, 1.1, 0.9, 11.0, 1000, 5),  # 11× jump
    ])
    result = analyse_data_quality(candles, "subgraph")
    assert result.passed is False
    assert any("jump" in w for w in result.warnings)


def test_quality_single_swap_warning():
    """More than 30% single-swap candles should trigger a warning."""
    candles = _make_candles([
        (i * 3_600_000, 1.0, 1.1, 0.9, 1.0, 100, 1)   # all single-swap
        for i in range(10)
    ])
    result = analyse_data_quality(candles, "subgraph")
    assert any("1 swap" in w for w in result.warnings)


def test_quality_gecko_notice():
    """GeckoTerminal source should always add a VWAP limitation notice."""
    candles = _make_candles([
        (0, 1.0, 1.1, 0.9, 1.0, 1000, 0),
        (3600000, 1.0, 1.1, 0.9, 1.0, 1000, 0),
    ])
    result = analyse_data_quality(candles, "gecko")
    assert any("GeckoTerminal" in w for w in result.warnings)


def test_quality_confidence_medium_one_warning():
    """One warning → medium confidence."""
    candles = _make_candles([
        (0,       1.0, 6.0, 1.0, 1.5, 1000, 5),  # spike
        (3600000, 1.5, 1.6, 1.4, 1.55, 800, 4),
    ])
    result = analyse_data_quality(candles, "subgraph")
    assert result.confidence == "medium"


def test_quality_confidence_low_two_warnings():
    """Two warnings → low confidence."""
    candles = _make_candles([
        (0,       1.0, 6.0, 1.0, 1.0,  1000, 1),   # spike + single-swap
        (3600000, 1.0, 6.0, 1.0, 11.0, 1000, 1),   # spike + jump + single-swap
    ])
    result = analyse_data_quality(candles, "subgraph")
    assert result.confidence == "low"


# ── Pipeline dispatch ─────────────────────────────────────────────────────────

def test_pipeline_dispatches_to_pulsex():
    """chain='pulsechain' should call pulsex.fetch."""
    with patch("data.sources.pulsex.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("pulsechain", "0xtoken", 3600)
        mock_fetch.assert_called_once_with("0xtoken", 3600, "auto")


def test_pipeline_dispatches_to_thegraph():
    """chain='ethereum' should call thegraph.fetch."""
    with patch("data.sources.thegraph.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("ethereum", "0xtoken", 3600, graph_api_key="testkey")
        mock_fetch.assert_called_once_with("ethereum", "0xtoken", 3600, "testkey")


def test_pipeline_dispatches_to_gecko():
    """chain='solana' should call gecko.fetch."""
    with patch("data.sources.gecko.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("solana", "", gecko_pair_address="0xpair", gecko_network="solana")
        mock_fetch.assert_called_once_with("solana", "0xpair", "hour", 1000)


def test_pipeline_raises_on_unknown_chain():
    with pytest.raises(ValueError, match="Unknown chain"):
        fetch_candles("moonbeam", "0xtoken")


def test_pipeline_raises_gecko_without_pair_address():
    with pytest.raises(ValueError, match="gecko_pair_address"):
        fetch_candles("solana", "")


# ── Yahoo Finance dispatch ─────────────────────────────────────────────────

def test_pipeline_dispatches_to_yahoo_gold():
    """chain='gold' should call yahoo.fetch with symbol GC=F."""
    with patch("data.sources.yahoo.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("gold", "", bucket_seconds=86400)
        mock_fetch.assert_called_once_with("GC=F", 86400)


def test_pipeline_dispatches_to_yahoo_spx500():
    """chain='spx500' should call yahoo.fetch with symbol SPY."""
    with patch("data.sources.yahoo.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("spx500", "", bucket_seconds=86400)
        mock_fetch.assert_called_once_with("SPY", 86400)


def test_pipeline_dispatches_to_yahoo_spx500_full():
    """chain='spx500_full' should call yahoo.fetch with symbol ^GSPC."""
    with patch("data.sources.yahoo.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("spx500_full", "", bucket_seconds=86400)
        mock_fetch.assert_called_once_with("^GSPC", 86400)


def test_pipeline_yahoo_bucket_override():
    """Yahoo chains with default_bucket=86400 should override the 14400 crypto default."""
    with patch("data.sources.yahoo.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        # Caller passes 14400 (crypto default) — pipeline should upgrade to 86400 for yahoo chains
        fetch_candles("gold", "", bucket_seconds=14400)
        mock_fetch.assert_called_once_with("GC=F", 86400)


def test_pipeline_yahoo_stocks_uses_token_address():
    """chain='stocks' with no fixed symbol should use ASSET_TOKEN_ADDRESS as ticker."""
    with patch("data.sources.yahoo.fetch") as mock_fetch:
        mock_fetch.return_value = ([], DataQualityResult(passed=True, confidence="high"))
        fetch_candles("stocks", "AAPL", bucket_seconds=86400)
        mock_fetch.assert_called_once_with("AAPL", 86400)


def test_pipeline_yahoo_raises_without_symbol():
    """chain='stocks' with no token address and no fixed symbol should raise ValueError."""
    with pytest.raises(ValueError, match="No ticker symbol"):
        fetch_candles("stocks", "", bucket_seconds=86400)
