"""
PulseX v1/v2 subgraph data source.

Searches PulseChain for the best trading pair for a given token address,
paginates all swap events from genesis, and returns VWAP candles.
No API key required.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from data.candles import Candle, build_candles_from_swaps
from data.chains import PULSEX_ENDPOINTS, PULSEX_STABLECOINS, WPLS
from data.quality import DataQualityResult, analyse_data_quality

logger = logging.getLogger(__name__)

PAGE_SIZE = 1000
DELAY_BETWEEN_PAGES = 0.15  # seconds — matches JS backtester pacing


@dataclass
class _PairInfo:
    version: str
    pair_id: str
    token0_id: str
    token1_id: str
    token0_symbol: str
    token1_symbol: str
    is_token0: bool
    volume_usd: float


def fetch(
    token_address: str,
    bucket_seconds: int,
    version_pref: str = "auto",
) -> tuple[list[Candle], DataQualityResult]:
    """
    Fetch VWAP candles for a PulseChain token from the PulseX subgraph.

    Args:
        token_address:  Token contract address (checksummed or lowercase).
        bucket_seconds: Candle bucket width in seconds (e.g. 3600 = 1 hour).
        version_pref:   "auto" tries pulsex then pulsexv2, or pass "pulsex"/"pulsexv2".

    Returns:
        (candles, quality_result)
    """
    versions = ["pulsex", "pulsexv2"] if version_pref == "auto" else [version_pref]
    pair = _find_best_pair(token_address.lower(), versions)

    logger.info(
        "PulseX %s — pair %s/%s (id: %s)",
        pair.version, pair.token0_symbol, pair.token1_symbol, pair.pair_id[:10],
    )

    swaps = _paginate_swaps(pair)
    if not swaps:
        raise ValueError(f"No swaps found for pair {pair.pair_id} on PulseX {pair.version}")

    candles = build_candles_from_swaps(swaps, bucket_seconds, pair.is_token0, is_v2=True)
    if not candles:
        raise ValueError("Could not build candles from swap data — the token may have zero volume.")

    quality = analyse_data_quality(candles, "subgraph")
    logger.info(
        "Built %d candles — quality: %s (%d warning(s))",
        len(candles), quality.confidence, len(quality.warnings),
    )
    return candles, quality


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_best_pair(token_address: str, versions: list[str]) -> _PairInfo:
    """
    Search PulseX v1 and/or v2 for all pairs containing the token.
    Rank candidates: stablecoin partner > WPLS partner > other, then by volumeUSD.
    """
    query = f"""{{
      asToken0: pairs(
        first: 10
        orderBy: volumeUSD
        orderDirection: desc
        where: {{ token0: "{token_address}" }}
      ) {{ id volumeUSD token0 {{ id symbol }} token1 {{ id symbol }} }}
      asToken1: pairs(
        first: 10
        orderBy: volumeUSD
        orderDirection: desc
        where: {{ token1: "{token_address}" }}
      ) {{ id volumeUSD token0 {{ id symbol }} token1 {{ id symbol }} }}
    }}"""

    best_pair: _PairInfo | None = None

    for version in versions:
        try:
            data = _subgraph_query(version, query)
        except Exception as exc:
            logger.warning("PulseX %s search failed: %s", version, exc)
            continue

        candidates = [
            {**p, "is_token0": True}  for p in (data.get("asToken0") or [])
        ] + [
            {**p, "is_token0": False} for p in (data.get("asToken1") or [])
        ]
        if not candidates:
            continue

        ranked = sorted(candidates, key=lambda p: _pair_rank(p), reverse=True)
        top = ranked[0]
        vol = float(top["volumeUSD"])
        if best_pair is None or vol > best_pair.volume_usd:
            best_pair = _PairInfo(
                version=version,
                pair_id=top["id"],
                token0_id=top["token0"]["id"].lower(),
                token1_id=top["token1"]["id"].lower(),
                token0_symbol=top["token0"]["symbol"],
                token1_symbol=top["token1"]["symbol"],
                is_token0=top["is_token0"],
                volume_usd=vol,
            )

    if best_pair is None:
        raise ValueError(
            f"Token {token_address} not found in any PulseX subgraph. "
            "Verify the token contract address (not the pair address)."
        )
    return best_pair


def _pair_rank(pair: dict) -> tuple[int, float]:
    """Return (partner_type_score, volumeUSD) for sorting."""
    is_token0 = pair["is_token0"]
    partner_addr = (pair["token1"]["id"] if is_token0 else pair["token0"]["id"]).lower()
    if partner_addr in PULSEX_STABLECOINS:
        score = 2
    elif partner_addr == WPLS:
        score = 1
    else:
        score = 0
    return (score, float(pair["volumeUSD"]))


def _paginate_swaps(pair: _PairInfo) -> list[dict]:
    """Paginate all swaps for a pair using timestamp_gt cursor."""
    all_swaps: list[dict] = []
    last_timestamp = 0

    while True:
        query = f"""{{
          swaps(
            first: {PAGE_SIZE}
            orderBy: timestamp
            orderDirection: asc
            where: {{
              pair: "{pair.pair_id}"
              timestamp_gt: "{last_timestamp}"
            }}
          ) {{
            timestamp
            amount0In
            amount0Out
            amount1In
            amount1Out
            amountUSD
          }}
        }}"""

        data = _subgraph_query(pair.version, query)
        swaps = data.get("swaps") or []
        if not swaps:
            break

        all_swaps.extend(swaps)
        last_timestamp = int(swaps[-1]["timestamp"])
        logger.debug("PulseX page fetched — %d swaps so far", len(all_swaps))

        if len(swaps) < PAGE_SIZE:
            break
        time.sleep(DELAY_BETWEEN_PAGES)

    return all_swaps


def _subgraph_query(version: str, query: str) -> dict:
    endpoint = PULSEX_ENDPOINTS[version]
    resp = requests.post(
        endpoint,
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"PulseX subgraph HTTP {resp.status_code} ({version})")
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"PulseX GraphQL error: {body['errors'][0].get('message')}")
    return body["data"]
