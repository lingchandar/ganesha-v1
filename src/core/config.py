"""
GANESHA V1 — Central Configuration Loader
Loads all environment variables from config/.env with validation.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class GaneshaConfig(BaseSettings):
    """
    Central configuration for the entire GANESHA V1 engine.
    All values are loaded from config/.env with type validation.
    """

    # ── Broker Selection ──
    broker_name: str = Field(default="fyers", description="Broker API: 'fyers' or 'angelone'")

    # ── Fyers API v3 Credentials ──
    fyers_app_id: str = Field(default="", description="Fyers developer App ID")
    fyers_secret_key: str = Field(default="", description="Fyers App secret key")
    fyers_redirect_uri: str = Field(default="https://127.0.0.1", description="OAuth2 redirect URI")

    # ── Angel One SmartAPI Credentials ──
    angel_api_key: Optional[str] = Field(default=None, description="Angel One API key")
    angel_client_id: Optional[str] = Field(default=None, description="Angel One client/login ID")
    angel_password: Optional[str] = Field(default=None, description="Angel One trading password")
    angel_totp_secret: Optional[str] = Field(default=None, description="Angel One TOTP seed")

    # ── PostgreSQL Database ──
    db_host: str = Field(default="localhost", description="Database hostname")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="ganesha_v1", description="Database name")
    db_user: str = Field(default="ganesha_admin", description="Database username")
    db_password: str = Field(default="", description="Database password")

    # ── Gemini AI ──
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Capital & Risk Parameters ──
    total_capital_inr: float = Field(default=500000.0, description="Total trading capital in INR")
    max_risk_per_trade_pct: float = Field(default=0.01, description="Max risk per trade as decimal (1% = 0.01)")
    max_concurrent_positions: int = Field(default=5, description="Max simultaneous open positions")
    max_stocks_per_sector: int = Field(default=2, description="Max stocks from same sector")

    # ── Rate Limiting (SEBI Compliance) ──
    api_min_interval_ms: int = Field(default=250, description="Minimum ms between API requests")
    api_max_requests_per_minute: int = Field(default=120, description="Max requests per rolling minute")

    @property
    def database_url(self) -> str:
        """Construct the SQLAlchemy-compatible PostgreSQL connection string."""
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def async_database_url(self) -> str:
        """Async variant of the database URL for use with asyncpg."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent.parent / "config" / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ── Singleton Configuration Instance ──
# Import this anywhere: from src.core.config import settings
settings = GaneshaConfig()
