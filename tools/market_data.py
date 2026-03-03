"""
MarketDataTool — exposes the data pipeline as a BaseTool so agents can
request historical candle data via the ToolRegistry.
"""
from __future__ import annotations

from dataclasses import asdict

from config.settings import GRAPH_API_KEY
from data.pipeline import fetch_candles
from tools.base import BaseTool, ToolResult


class MarketDataTool(BaseTool):
    name = "fetch_market_data"
    description = (
        "Fetch historical OHLCV candle data for a crypto token. "
        "Supports PulseChain (PulseX subgraph), Ethereum/Arbitrum/Base/Polygon/BSC/Avalanche "
        "(Uniswap v3 via The Graph), and Solana (GeckoTerminal fallback). "
        "Returns candles with open, high, low, close (VWAP), volume, and swap_count fields, "
        "plus a data quality assessment."
    )

    def run(
        self,
        chain: str,
        token_address: str = "",
        bucket_seconds: int = 3600,
        gecko_pair_address: str = "",
        gecko_network: str = "",
        gecko_timeframe: str = "hour",
        gecko_limit: int = 1000,
        pulsex_version_pref: str = "auto",
        **_kwargs,
    ) -> ToolResult:
        try:
            candles, quality = fetch_candles(
                chain=chain,
                token_address=token_address,
                bucket_seconds=bucket_seconds,
                graph_api_key=GRAPH_API_KEY,
                gecko_pair_address=gecko_pair_address,
                gecko_network=gecko_network,
                gecko_timeframe=gecko_timeframe,
                gecko_limit=gecko_limit,
                pulsex_version_pref=pulsex_version_pref,
            )
            return ToolResult(
                success=True,
                output={
                    "candle_count": len(candles),
                    "candles": [asdict(c) for c in candles],
                    "quality": asdict(quality),
                },
            )
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chain": {
                        "type": "string",
                        "description": (
                            "Blockchain to fetch data from. "
                            "Options: pulsechain, ethereum, arbitrum, base, polygon, bsc, avalanche, solana"
                        ),
                        "enum": [
                            "pulsechain", "ethereum", "arbitrum", "base",
                            "polygon", "bsc", "avalanche", "solana",
                        ],
                    },
                    "token_address": {
                        "type": "string",
                        "description": "Token contract address. Required for all chains except solana.",
                    },
                    "bucket_seconds": {
                        "type": "integer",
                        "description": "Candle width in seconds. Default 3600 (1 hour). Use 86400 for daily.",
                        "default": 3600,
                    },
                    "gecko_pair_address": {
                        "type": "string",
                        "description": "Pool/pair contract address. Required for solana (GeckoTerminal source).",
                    },
                    "gecko_timeframe": {
                        "type": "string",
                        "description": "GeckoTerminal timeframe: 'minute', 'hour', or 'day'. Default 'hour'.",
                        "enum": ["minute", "hour", "day"],
                        "default": "hour",
                    },
                    "gecko_limit": {
                        "type": "integer",
                        "description": "Max candles to fetch from GeckoTerminal. Default 1000.",
                        "default": 1000,
                    },
                },
                "required": ["chain"],
            },
        }
