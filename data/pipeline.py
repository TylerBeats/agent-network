"""
Top-level data pipeline dispatcher.

Routes fetch_candles() calls to the correct source module based on CHAIN_CONFIG.
This is the single entry point all agents and tools use to obtain candle data.
"""
from __future__ import annotations

import logging

from data.candles import Candle
from data.chains import CHAIN_CONFIG
from data.quality import DataQualityResult, trim_low_liquidity_candles

logger = logging.getLogger(__name__)


def fetch_candles(
    chain: str,
    token_address: str,
    bucket_seconds: int = 3600,
    graph_api_key: str = "",
    gecko_pair_address: str = "",
    gecko_network: str = "",
    gecko_timeframe: str = "hour",
    gecko_limit: int = 1000,
    pulsex_version_pref: str = "auto",
) -> tuple[list[Candle], DataQualityResult]:
    """
    Fetch historical OHLCV candles for a crypto token.

    Dispatches to the correct source based on CHAIN_CONFIG[chain]["src"]:
      "subgraph" -> data/sources/pulsex.py  (PulseChain, no API key)
      "thegraph"  -> data/sources/thegraph.py (Ethereum/Arbitrum/Base/Polygon/BSC/Avalanche)
      "gecko"     -> data/sources/gecko.py   (Solana + chains without subgraph)

    Args:
        chain:               Chain name key from CHAIN_CONFIG (e.g. "pulsechain", "ethereum").
        token_address:       Token contract address. Not required for "gecko" source.
        bucket_seconds:      Candle width in seconds. Default 3600 (1 hour).
        graph_api_key:       Required for "thegraph" source.
        gecko_pair_address:  Required for "gecko" source -- the pool/pair address.
        gecko_network:       GeckoTerminal network slug. Defaults to CHAIN_CONFIG value.
        gecko_timeframe:     "minute", "hour", or "day". Default "hour".
        gecko_limit:         Max candles to fetch from GeckoTerminal. Default 1000.
        pulsex_version_pref: "auto", "pulsex", or "pulsexv2". Default "auto".

    Returns:
        (candles, DataQualityResult)
    """
    if chain not in CHAIN_CONFIG:
        supported = ", ".join(CHAIN_CONFIG.keys())
        raise ValueError(f"Unknown chain '{chain}'. Supported: {supported}")

    cfg = CHAIN_CONFIG[chain]
    src = cfg["src"]

    # For Yahoo Finance chains, prefer the chain's default bucket if caller passed
    # the global default (14400). This avoids requesting 4h bars for daily-only markets.
    effective_bucket = bucket_seconds
    if src == "yahoo" and "default_bucket" in cfg and bucket_seconds == 14400:
        effective_bucket = cfg["default_bucket"]
        if effective_bucket != bucket_seconds:
            logger.info(
                "fetch_candles: using chain default bucket %ds for %s (was %ds)",
                effective_bucket, chain, bucket_seconds,
            )

    logger.info("fetch_candles: chain=%s src=%s bucket=%ds", chain, src, effective_bucket)

    if src == "subgraph":
        from data.sources import pulsex
        candles, quality = pulsex.fetch(token_address, effective_bucket, pulsex_version_pref)
    elif src == "thegraph":
        from data.sources import thegraph
        candles, quality = thegraph.fetch(chain, token_address, effective_bucket, graph_api_key)
    elif src == "gecko":
        from data.sources import gecko
        network = gecko_network or cfg.get("gecko", chain)
        if not gecko_pair_address:
            raise ValueError(
                f"gecko_pair_address is required for chain '{chain}' (GeckoTerminal source). "
                "Pass the pool/pair contract address."
            )
        candles, quality = gecko.fetch(network, gecko_pair_address, gecko_timeframe, gecko_limit)
    elif src == "yahoo":
        from data.sources import yahoo
        # symbol is fixed in CHAIN_CONFIG, or user-supplied via token_address
        symbol = cfg.get("symbol") or token_address
        if not symbol:
            raise ValueError(
                f"No ticker symbol configured for chain '{chain}'. "
                "Set ASSET_TOKEN_ADDRESS to a Yahoo Finance ticker (e.g. 'SPY', 'GLD')."
            )
        candles, quality = yahoo.fetch(symbol, effective_bucket)
    else:
        raise ValueError(f"Unrecognised data source '{src}' for chain '{chain}'")

    # Apply genesis liquidity filter -- remove early low-liquidity candles from crypto sources.
    # Skipped for Yahoo Finance: volume is in shares/contracts (not USD), so the USD threshold
    # is meaningless, and established markets (SPY, GC=F, QQQ, ^GSPC) are liquid from day 1.
    if src != "yahoo":
        candles = trim_low_liquidity_candles(candles, bucket_seconds=effective_bucket)

    return candles, quality
