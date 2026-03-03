import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
DEFAULT_MODEL: str     = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")
MAX_STEPS: int         = int(os.getenv("MAX_STEPS", "10"))
LOG_LEVEL: str         = os.getenv("LOG_LEVEL", "INFO")
GRAPH_API_KEY: str     = os.getenv("GRAPH_API_KEY", "")

# Broker / execution settings
ALPACA_API_KEY: str    = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
BROKER_MODE: str       = os.getenv("BROKER_MODE", "dry_run")   # "dry_run" | "alpaca"

# Asset configuration
ASSET_CHAIN: str          = os.getenv("ASSET_CHAIN", "pulsechain")
ASSET_TOKEN_ADDRESS: str  = os.getenv("ASSET_TOKEN_ADDRESS", "")
ASSET_BUCKET_SECONDS: int = int(os.getenv("ASSET_BUCKET_SECONDS", "14400"))  # 4h default
