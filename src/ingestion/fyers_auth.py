"""
GANESHA V1 — Fyers API v3 Authentication Manager (Module 1, 2)

SEBI COMPLIANCE & CREDENTIAL SECURITY:
This module manages the Fyers OAuth2 authorization code flow.
Tokens are saved locally in config/fyers_token.json (ignored by Git).
Once authenticated for the day, the access token remains valid until 06:00 AM IST next morning.

Usage:
    Run as script when ready with credentials:
    .venv/bin/python -m src.ingestion.fyers_auth
"""
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from src.core.config import settings


TOKEN_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "fyers_token.json"


class FyersAuthManager:
    """
    Handles Fyers OAuth2 login, token generation, caching, and validation.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self.app_id = app_id or settings.fyers_app_id
        self.secret_key = secret_key or settings.fyers_secret_key
        self.redirect_uri = redirect_uri or settings.fyers_redirect_uri

    def get_login_url(self) -> str:
        """
        Generate the Fyers authorization URL for the user to visit in their browser.
        """
        if not self.app_id or not self.secret_key:
            raise ValueError(
                "FYERS_APP_ID and FYERS_SECRET_KEY must be configured in config/.env before generating auth URL."
            )

        from fyers_apiv3 import fyersModel

        session = fyersModel.SessionModel(
            client_id=self.app_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        auth_url = session.generate_authcode()
        return auth_url

    def exchange_code_for_token(self, auth_code_or_url: str) -> str:
        """
        Exchanges the authorization code or full redirected URL for a daily access token.
        Saves the token to config/fyers_token.json.
        """
        from fyers_apiv3 import fyersModel
        import urllib.parse

        # Extract auth_code if full URL is pasted
        code = auth_code_or_url.strip()
        if "auth_code=" in code:
            parsed = urllib.parse.urlparse(code)
            params = urllib.parse.parse_qs(parsed.query)
            code = params.get("auth_code", [code])[0]

        session = fyersModel.SessionModel(
            client_id=self.app_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        session.set_token(code)
        response = session.generate_token()

        if response.get("s") == "ok" and "access_token" in response:
            token = response["access_token"]
            self._save_token(token, response)
            logger.success("Fyers API v3 authentication successful! Token saved.")
            return token
        else:
            err = response.get("message", "Unknown error")
            logger.error(f"Fyers token generation failed: {err}")
            raise RuntimeError(f"FYERS_AUTH_ERROR: {err}")

    def _save_token(self, token: str, raw_response: Dict) -> None:
        """Save access token and timestamp to config/fyers_token.json."""
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "access_token": token,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "trade_date": date.today().isoformat(),
            "app_id": self.app_id,
            "raw_response": raw_response,
        }
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_cached_token(cls) -> Optional[str]:
        """
        Load cached token if exists and was generated for the current calendar date.
        """
        if not TOKEN_FILE.exists():
            return None

        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if token is from today
            token_date = data.get("trade_date")
            if token_date == date.today().isoformat():
                return data.get("access_token")
            else:
                logger.info("Cached Fyers token has expired (new trading session). Re-auth required.")
                return None
        except Exception as e:
            logger.warning(f"Failed to read cached token: {e}")
            return None


def interactive_login():
    """CLI interactive helper for Fyers authentication."""
    print("=" * 70)
    print("🕉️ GANESHA V1 — FYERS API v3 INTERACTIVE AUTHENTICATION")
    print("=" * 70)

    auth = FyersAuthManager()
    try:
        url = auth.get_login_url()
        print(f"\n1. Open this URL in your browser and log in to FYERS:")
        print(f"\n   {url}\n")
        print("2. After login, you will be redirected to your redirect URI.")
        print("   Copy the FULL redirected URL from your browser address bar and paste it below:\n")

        user_input = input("Paste redirected URL (or auth_code): ").strip()
        token = auth.exchange_code_for_token(user_input)
        print("\n✅ Authentication Successful! You are ready to scan.")
    except Exception as e:
        print(f"\n❌ Error during authentication: {e}")


if __name__ == "__main__":
    interactive_login()
