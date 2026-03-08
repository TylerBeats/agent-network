# Chain configuration, endpoints, and token address constants.
# All addresses are lowercase to match subgraph query requirements.

# ── PulseChain ────────────────────────────────────────────────────────────────

WPLS = "0xa1077a294dde1b09bb078844df40758a5d0f9a27"

PULSEX_STABLECOINS: frozenset[str] = frozenset({
    "0xefd766ccb38eaf1dfd701853bfce31359239f305",  # DAI from Ethereum
    "0x15d38573d2feeb82e7ad5187ab8c1d52810b1f07",  # USDC
    "0x0cb6f5a34ad42ec934882a05265a7d5f59b51a2f",  # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # bridged USDC
})

PULSEX_ENDPOINTS: dict[str, str] = {
    "pulsex":   "https://graph.pulsechain.com/subgraphs/name/pulsechain/pulsex",
    "pulsexv2": "https://graph.pulsechain.com/subgraphs/name/pulsechain/pulsexv2",
}

# ── The Graph (Uniswap v3) ────────────────────────────────────────────────────

THEGRAPH_GATEWAY = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"

UNISWAP_SUBGRAPH_IDS: dict[str, str] = {
    "ethereum":  "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
    "arbitrum":  "FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM",
    "base":      "43Hwfi3dJSoGpyas9VwNoDAv55yjgGrPpNSmbQZArzMG",
    "polygon":   "3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm",
    "bsc":       "F85MNzUGYqgSHSHRGgeVMNsdnW1KtZSVgFULumXRZTw2",
    "avalanche": "GVH9h9KZ9CqheUEL93qMbq7QwgoBu32QXQDPR6bev4Eo",
}

# Per-chain stablecoin whitelists (for pair ranking)
CHAIN_STABLES: dict[str, frozenset[str]] = {
    "ethereum": frozenset({
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
    }),
    "arbitrum": frozenset({
        "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",  # USDC.e
        "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",  # USDT
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",  # DAI
    }),
    "base": frozenset({
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",  # USDC
        "0x50c5725949a6f0c72e6c4a641f24049a917db0cb",  # DAI
    }),
    "polygon": frozenset({
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",  # USDC
        "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",  # USDT
        "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063",  # DAI
    }),
    "bsc": frozenset({
        "0x55d398326f99059ff775485246999027b3197955",  # USDT
        "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD
        "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC
    }),
    "avalanche": frozenset({
        "0xb97ef9ef8734c71904d8002f8b6bc66dd9c48a6e",  # USDC
        "0x9702230a8ea53601f5cd2dc00fdbc13d4df4a8c7",  # USDT
        "0xd586e7f844cea2f87f50152665bcbc2c279d8d70",  # DAI
    }),
}

# Per-chain wrapped native tokens (for pair ranking fallback)
CHAIN_WNATIVE: dict[str, str] = {
    "ethereum":  "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "arbitrum":  "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",  # WETH on Arbitrum
    "base":      "0x4200000000000000000000000000000000000006",  # WETH on Base
    "polygon":   "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",  # WMATIC
    "bsc":       "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",  # WBNB
    "avalanche": "0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7",  # WAVAX
}

# ── Chain routing config ──────────────────────────────────────────────────────
# "src" tells the pipeline which data source module to use:
#   "subgraph" → data/sources/pulsex.py
#   "thegraph" → data/sources/thegraph.py
#   "gecko"    → data/sources/gecko.py

CHAIN_CONFIG: dict[str, dict] = {
    "pulsechain": {
        "src":   "subgraph",
        "label": "PulseChain",
        "dex":   "PulseX v1/v2",
        "gecko": "pulsechain",
        "note":  "On-chain subgraph · Full history from genesis · No API key required",
    },
    "ethereum": {
        "src":          "thegraph",
        "label":        "Ethereum",
        "dex":          "Uniswap v3",
        "gecko":        "eth",
        "note":         "The Graph · Uniswap v3 · Full history — Graph API key required",
    },
    "arbitrum": {
        "src":          "thegraph",
        "label":        "Arbitrum",
        "dex":          "Uniswap v3",
        "gecko":        "arbitrum",
        "note":         "The Graph · Uniswap v3 · Full history — Graph API key required",
    },
    "base": {
        "src":          "thegraph",
        "label":        "Base",
        "dex":          "Uniswap v3",
        "gecko":        "base",
        "note":         "The Graph · Uniswap v3 · Full history — Graph API key required",
    },
    "polygon": {
        "src":          "thegraph",
        "label":        "Polygon",
        "dex":          "Uniswap v3",
        "gecko":        "polygon_pos",
        "note":         "The Graph · Uniswap v3 · Full history — Graph API key required",
    },
    "bsc": {
        "src":          "thegraph",
        "label":        "BSC",
        "dex":          "Uniswap v3 / PancakeSwap",
        "gecko":        "bsc",
        "note":         "The Graph · Uniswap v3 · Full history — Graph API key required",
    },
    "avalanche": {
        "src":          "thegraph",
        "label":        "Avalanche",
        "dex":          "Uniswap v3",
        "gecko":        "avax",
        "note":         "The Graph · Uniswap v3 · Full history — Graph API key required",
    },
    "solana": {
        "src":   "gecko",
        "label": "Solana",
        "dex":   "Raydium / Orca",
        "gecko": "solana",
        "note":  "GeckoTerminal OHLCV · ~3 months hourly history · No subgraph available",
    },

    # ── Traditional markets via Yahoo Finance ─────────────────────────────────
    # No API key required. Close prices are split- and dividend-adjusted.
    # Volume is in shares/contracts (valid for relative VOLUME_ABOVE_SMA signals).
    # Recommended ASSET_BUCKET_SECONDS=86400 (daily bars) for these markets.

    "gold": {
        "src":            "yahoo",
        "label":          "Gold (GC=F Futures)",
        "symbol":         "GC=F",   # Gold futures continuous contract — daily OHLCV from ~1983
        "default_bucket": 86400,
        "note":           "Yahoo Finance · Gold Futures (GC=F) · Daily OHLCV · Full history ~1983 · No API key",
    },
    "spx500": {
        "src":            "yahoo",
        "label":          "S&P 500 (SPY ETF)",
        "symbol":         "SPY",    # SPDR S&P 500 — most liquid ETF on earth; has real volume
        "default_bucket": 86400,
        "note":           "Yahoo Finance · SPY ETF · Daily OHLCV · Full history ~1993 · No API key",
    },
    "spx500_full": {
        "src":            "yahoo",
        "label":          "S&P 500 Full History (^GSPC)",
        "symbol":         "^GSPC",  # S&P 500 index — daily from ~1927; volume is synthetic, avoid VOLUME_ABOVE_SMA
        "default_bucket": 86400,
        "note":           "Yahoo Finance · ^GSPC index · Daily OHLCV · Full history ~1927 · No API key · Note: volume is synthetic",
    },
    "nasdaq": {
        "src":            "yahoo",
        "label":          "Nasdaq-100 (QQQ ETF)",
        "symbol":         "QQQ",    # Invesco Nasdaq-100 Trust
        "default_bucket": 86400,
        "note":           "Yahoo Finance · QQQ ETF · Daily OHLCV · Full history ~1999 · No API key",
    },
    "stocks": {
        "src":            "yahoo",
        "label":          "US Stocks / ETFs",
        "symbol":         "",       # set ASSET_TOKEN_ADDRESS to the Yahoo ticker (e.g. AAPL, MSFT)
        "default_bucket": 86400,
        "note":           "Yahoo Finance · Any US equity or ETF · Daily OHLCV · No API key",
    },
    "crypto_yahoo": {
        "src":            "yahoo",
        "label":          "Crypto (Yahoo Finance)",
        "symbol":         "",       # set ASSET_TOKEN_ADDRESS to e.g. BTC-USD, ETH-USD
        "default_bucket": 86400,
        "note":           "Yahoo Finance · Major crypto daily OHLCV · Useful for BTC-USD / ETH-USD",
    },
    "nas100": {
        "src":            "yahoo",
        "label":          "Nasdaq-100 (QQQ ETF)",
        "symbol":         "QQQ",
        "default_bucket": 86400,
        "note":           "Yahoo Finance - QQQ ETF - Daily OHLCV - Full history ~1999 - No API key",
    },
    "us30": {
        "src":            "yahoo",
        "label":          "Dow Jones Industrial Average (DIA ETF)",
        "symbol":         "DIA",
        "default_bucket": 86400,
        "note":           "Yahoo Finance - DIA ETF - Daily OHLCV - Full history ~1998 - No API key",
    },
    "xauusd": {
        "src":            "yahoo",
        "label":          "Gold / USD (GLD ETF)",
        "symbol":         "GLD",
        "default_bucket": 86400,
        "note":           "Yahoo Finance - GLD ETF - Daily OHLCV - Full history ~2004 - No API key",
    },
    "spx500_1m":  {"src": "yahoo", "label": "SPX 1min",  "symbol": "SPY", "default_bucket": 60},
    "spx500_5m":  {"src": "yahoo", "label": "SPX 5min",  "symbol": "SPY", "default_bucket": 300},
    "spx500_15m": {"src": "yahoo", "label": "SPX 15min", "symbol": "SPY", "default_bucket": 900},
    "spx500_30m": {"src": "yahoo", "label": "SPX 30min", "symbol": "SPY", "default_bucket": 1800},
    "spx500_1h":  {"src": "yahoo", "label": "SPX 1hr",   "symbol": "SPY", "default_bucket": 3600},
    "spx500_4h":  {"src": "yahoo", "label": "SPX 4hr",   "symbol": "SPY", "default_bucket": 14400},
    "nas100_1m":  {"src": "yahoo", "label": "NAS 1min",  "symbol": "QQQ", "default_bucket": 60},
    "nas100_5m":  {"src": "yahoo", "label": "NAS 5min",  "symbol": "QQQ", "default_bucket": 300},
    "nas100_15m": {"src": "yahoo", "label": "NAS 15min", "symbol": "QQQ", "default_bucket": 900},
    "nas100_30m": {"src": "yahoo", "label": "NAS 30min", "symbol": "QQQ", "default_bucket": 1800},
    "nas100_1h":  {"src": "yahoo", "label": "NAS 1hr",   "symbol": "QQQ", "default_bucket": 3600},
    "nas100_4h":  {"src": "yahoo", "label": "NAS 4hr",   "symbol": "QQQ", "default_bucket": 14400},
    "us30_1m":    {"src": "yahoo", "label": "US30 1min", "symbol": "DIA", "default_bucket": 60},
    "us30_5m":    {"src": "yahoo", "label": "US30 5min", "symbol": "DIA", "default_bucket": 300},
    "us30_15m":   {"src": "yahoo", "label": "US30 15min","symbol": "DIA", "default_bucket": 900},
    "us30_30m":   {"src": "yahoo", "label": "US30 30min","symbol": "DIA", "default_bucket": 1800},
    "us30_1h":    {"src": "yahoo", "label": "US30 1hr",  "symbol": "DIA", "default_bucket": 3600},
    "us30_4h":    {"src": "yahoo", "label": "US30 4hr",  "symbol": "DIA", "default_bucket": 14400},
    "xauusd_1m":  {"src": "yahoo", "label": "XAU 1min",  "symbol": "GLD", "default_bucket": 60},
    "xauusd_5m":  {"src": "yahoo", "label": "XAU 5min",  "symbol": "GLD", "default_bucket": 300},
    "xauusd_15m": {"src": "yahoo", "label": "XAU 15min", "symbol": "GLD", "default_bucket": 900},
    "xauusd_30m": {"src": "yahoo", "label": "XAU 30min", "symbol": "GLD", "default_bucket": 1800},
    "xauusd_1h":  {"src": "yahoo", "label": "XAU 1hr",   "symbol": "GLD", "default_bucket": 3600},
    "xauusd_4h":  {"src": "yahoo", "label": "XAU 4hr",   "symbol": "GLD", "default_bucket": 14400},
}