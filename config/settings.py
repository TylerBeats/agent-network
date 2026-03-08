import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "local-only")
DEFAULT_MODEL: str     = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")
MAX_STEPS: int         = int(os.getenv("MAX_STEPS", "10"))
LOG_LEVEL: str         = os.getenv("LOG_LEVEL", "INFO")
GRAPH_API_KEY: str     = os.getenv("GRAPH_API_KEY", "")

ALPACA_API_KEY: str    = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
BROKER_MODE: str       = os.getenv("BROKER_MODE", "dry_run")

ASSET_CHAIN: str          = os.getenv("ASSET_CHAIN", "pulsechain")
ASSET_TOKEN_ADDRESS: str  = os.getenv("ASSET_TOKEN_ADDRESS", "")
ASSET_BUCKET_SECONDS: int = int(os.getenv("ASSET_BUCKET_SECONDS", "14400"))

LOCAL_LLM: bool          = os.getenv("LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_MODEL: str     = os.getenv("LOCAL_LLM_MODEL", "")
LOCAL_LLM_BASE_URL: str  = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
