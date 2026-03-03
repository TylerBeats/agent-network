"""
Uniswap v3 via The Graph decentralized network.

Supports Ethereum, Arbitrum, Base, Polygon, BSC, and Avalanche.
Requires a free Graph API key from thegraph.com/studio/apikeys.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from data.candles import Candle, build_candles_from_swaps
from data.chains import (
    CHAIN_STABLES,
    CHAIN_WNATIVE,
    THEGRAPH_GATEWAY,
    UNISWAP_SUBGRAPH_IDS,
)
from data.quality import DataQualityResult, analyse_data_quality

logger = logging.getLogger(__name__)

PAGE_SIZE = 1000
DELAY_BETWEEN_PAGES = 0.15  # seconds


@dataclass
class _PoolInfo:
    pool_id: str
    token0_id: str
    token1_id: str
    token0_symbol: str
    token1_symbol: str
    is_token0: bool
    tvl_usd: float
    fee_tier: int


def fetch(
    chain: str,
    token_address: str,
    bucket_seconds: int,
    api_key: str,
) -> tuple[list[Candle], DataQualityResult]:
    """
    Fetch VWAP candles for a token via Uniswap v3 on The Graph.

    Args:
        chain:          One of: ethereum, arbitrum, base, polygon, bsc, avalanche.
        token_address:  Token contract address.
        bucket_seconds: Candle bucket width in seconds.
        api_key:        The Graph API key.

    Returns:
        (candles, quality_result)
    """
    if chain not in UNISWAP_SUBGRAPH_IDS:
        raise ValueError(f"No Uniswap v3 subgraph configured for chain '{chain}'")
    if not api_key:
        raise ValueError("A Graph API key is required for Uniswap v3 chains. See thegraph.com/studio/apikeys")

    subgraph_id = UNISWAP_SUBGRAPH_IDS[chain]
    pool = _find_best_pool(token_address.lower(), chain, subgraph_id, api_key)

    our_symbol     = pool.token0_symbol if pool.is_token0 else pool.token1_symbol
    partner_symbol = pool.token1_symbol if pool.is_token0 else pool.token0_symbol
    fee_pct = pool.fee_tier / 10_000
    logger.info(
        "Uniswap v3 (%s) — %s/%s %.2f%% fee (pool: %s)",
        chain, our_symbol, partner_symbol, fee_pct, pool.pool_id[:10],
    )

    swaps = _paginate_swaps(pool, subgraph_id, api_key)
    if not swaps:
        raise ValueError(f"No swaps found for pool {pool.pool_id} on {chain}")

    candles = build_candles_from_swaps(swaps, bucket_seconds, pool.is_token0, is_v2=False)
    if not candles:
        raise ValueError("Could not build candles — the pool may have no swap history.")

    quality = analyse_data_quality(candles, "subgraph")
    logger.info(
        "Built %d candles — quality: %s (%d warning(s))",
        len(candles), quality.confidence, len(quality.warnings),
    )
    return candles, quality


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_best_pool(
    token_address: str,
    chain: str,
    subgraph_id: str,
    api_key: str,
) -> _PoolInfo:
    """
    Search for all Uniswap v3 pools containing the token.
    Rank: stablecoin partner > wrapped native > other, then by totalValueLockedUSD.
    """
    query = f"""{{
      asToken0: pools(
        first: 10
        orderBy: totalValueLockedUSD
        orderDirection: desc
        where: {{ token0: "{token_address}" }}
      ) {{ id totalValueLockedUSD feeTier token0 {{ id symbol }} token1 {{ id symbol }} }}
      asToken1: pools(
        first: 10
        orderBy: totalValueLockedUSD
        orderDirection: desc
        where: {{ token1: "{token_address}" }}
      ) {{ id totalValueLockedUSD feeTier token0 {{ id symbol }} token1 {{ id symbol }} }}
    }}"""

    data = _thegraph_query(subgraph_id, api_key, query)
    stables = CHAIN_STABLES.get(chain, frozenset())
    wnative = CHAIN_WNATIVE.get(chain, "")

    candidates = [
        {**p, "is_token0": True}  for p in (data.get("asToken0") or [])
    ] + [
        {**p, "is_token0": False} for p in (data.get("asToken1") or [])
    ]
    if not candidates:
        raise ValueError(
            f"Token {token_address} not found in Uniswap v3 on {chain}. "
            "Verify the token contract address."
        )

    ranked = sorted(candidates, key=lambda p: _pool_rank(p, stables, wnative), reverse=True)
    top = ranked[0]
    return _PoolInfo(
        pool_id=top["id"],
        token0_id=top["token0"]["id"].lower(),
        token1_id=top["token1"]["id"].lower(),
        token0_symbol=top["token0"]["symbol"],
        token1_symbol=top["token1"]["symbol"],
        is_token0=top["is_token0"],
        tvl_usd=float(top["totalValueLockedUSD"]),
        fee_tier=int(top["feeTier"]),
    )


def _pool_rank(pool: dict, stables: frozenset, wnative: str) -> tuple[int, float]:
    is_token0 = pool["is_token0"]
    partner_addr = (pool["token1"]["id"] if is_token0 else pool["token0"]["id"]).lower()
    if partner_addr in stables:
        score = 2
    elif partner_addr == wnative:
        score = 1
    else:
        score = 0
    return (score, float(pool["totalValueLockedUSD"]))


def _paginate_swaps(pool: _PoolInfo, subgraph_id: str, api_key: str) -> list[dict]:
    """Paginate all swaps for a pool using timestamp_gt cursor."""
    all_swaps: list[dict] = []
    last_timestamp = 0

    while True:
        query = f"""{{
          swaps(
            first: {PAGE_SIZE}
            orderBy: timestamp
            orderDirection: asc
            where: {{ pool: "{pool.pool_id}", timestamp_gt: "{last_timestamp}" }}
          ) {{
            timestamp
            amount0
            amount1
            amountUSD
          }}
        }}"""

        data = _thegraph_query(subgraph_id, api_key, query)
        swaps = data.get("swaps") or []
        if not swaps:
            break

        all_swaps.extend(swaps)
        last_timestamp = int(swaps[-1]["timestamp"])
        logger.debug("The Graph page fetched — %d swaps so far", len(all_swaps))

        if len(swaps) < PAGE_SIZE:
            break
        time.sleep(DELAY_BETWEEN_PAGES)

    return all_swaps


def _thegraph_query(subgraph_id: str, api_key: str, query: str) -> dict:
    endpoint = THEGRAPH_GATEWAY.format(api_key=api_key, subgraph_id=subgraph_id)
    resp = requests.post(
        endpoint,
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code in (401, 403):
        raise RuntimeError(
            "The Graph API key is invalid or expired. "
            "Check your key at thegraph.com/studio/apikeys"
        )
    if not resp.ok:
        raise RuntimeError(f"The Graph HTTP {resp.status_code}: {resp.text[:200]}")
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"The Graph GraphQL error: {body['errors'][0].get('message')}")
    return body["data"]
