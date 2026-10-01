"""Real-time FYERS market-data WebSocket client for Ganesha v1.

This module is deliberately read-only. It consumes FYERS SymbolUpdate
messages and normalizes them into Ganesha's internal live-tick structure.
It never exposes broker order operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Iterable, Optional

from loguru import logger

from src.core.config import settings
from src.ingestion.fyers_auth import FyersAuthManager


@dataclass(frozen=True)
class LiveMarketTick:
    """Normalized read-only market update from FYERS."""

    symbol: str
    ltp: Optional[float]
    open_price: Optional[float]
    high_price: Optional[float]
    low_price: Optional[float]
    previous_close: Optional[float]
    volume_traded: Optional[int]
    timestamp: Optional[datetime]
    raw_message: dict


TickHandler = Callable[[LiveMarketTick], None]
EventHandler = Callable[[dict], None]


class FyersRealtimeClient:
    """Read-only FYERS Data WebSocket client.

    The SDK owns the WebSocket connection and its reconnect loop. This class
    owns the desired symbol set, message normalization, and application
    callbacks.
    """

    DATA_TYPE = "SymbolUpdate"

    def __init__(
        self,
        access_token: Optional[str] = None,
        tick_handler: Optional[TickHandler] = None,
        error_handler: Optional[EventHandler] = None,
        close_handler: Optional[EventHandler] = None,
    ) -> None:
        self._access_token = access_token or FyersAuthManager.load_cached_token()
        self._tick_handler = tick_handler
        self._error_handler = error_handler
        self._close_handler = close_handler
        self._symbols: set[str] = set()
        self._socket = None
        self._connected = False
        self._lock = RLock()

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the currently requested symbols in deterministic order."""
        with self._lock:
            return tuple(sorted(self._symbols))

    @property
    def connected(self) -> bool:
        return self._connected

    def _require_token(self) -> str:
        if not self._access_token:
            raise RuntimeError(
                "No valid FYERS access token available. "
                "Run 'python -m src.ingestion.fyers_auth' first."
            )
        return self._access_token

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_timestamp(value) -> Optional[datetime]:
        if value is None:
            return None
        try:
            # FYERS timestamps are Unix epoch seconds.
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None

    @classmethod
    def normalize_message(cls, message: dict) -> LiveMarketTick:
        """Convert a FYERS SymbolUpdate dictionary to a stable internal tick."""
        if not isinstance(message, dict):
            raise ValueError("FYERS WebSocket message must be a dictionary")

        symbol = str(message.get("symbol") or "").strip()
        if not symbol:
            raise ValueError("FYERS WebSocket message has no symbol")

        return LiveMarketTick(
            symbol=symbol,
            ltp=cls._to_float(message.get("ltp")),
            open_price=cls._to_float(message.get("open_price")),
            high_price=cls._to_float(message.get("high_price")),
            low_price=cls._to_float(message.get("low_price")),
            previous_close=cls._to_float(message.get("prev_close_price")),
            volume_traded=cls._to_int(message.get("vol_traded_today")),
            timestamp=cls._to_timestamp(message.get("last_traded_time")),
            raw_message=dict(message),
        )

    def _on_connect(self) -> None:
        with self._lock:
            self._connected = True
            symbols = list(sorted(self._symbols))

        if symbols:
            self._socket.subscribe(symbols=symbols, data_type=self.DATA_TYPE)
            logger.info("FYERS WebSocket subscribed to {} symbols", len(symbols))
        else:
            logger.warning("FYERS WebSocket connected with no symbols subscribed")

    def _on_message(self, message: dict) -> None:
        # FYERS can emit connection/subscription acknowledgements as well as
        # symbol updates. Only normalize messages that actually contain a symbol.
        if not isinstance(message, dict) or not message.get("symbol"):
            return

        try:
            tick = self.normalize_message(message)
            if self._tick_handler:
                self._tick_handler(tick)
        except Exception as exc:
            logger.warning("Ignoring invalid FYERS market-data message: {}", exc)

    def _on_error(self, message: dict) -> None:
        logger.error("FYERS WebSocket error: {}", message)
        if self._error_handler:
            self._error_handler(message)

    def _on_close(self, message: dict) -> None:
        with self._lock:
            self._connected = False
        logger.warning("FYERS WebSocket closed: {}", message)
        if self._close_handler:
            self._close_handler(message)

    def connect(self, symbols: Iterable[str] = ()) -> None:
        """Connect and begin the live SymbolUpdate stream."""
        requested = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
        with self._lock:
            self._symbols.update(requested)

        token = self._require_token()

        try:
            from fyers_apiv3.FyersWebsocket import data_ws
        except ImportError as exc:
            raise RuntimeError(
                "fyers-apiv3 is required for real-time market data."
            ) from exc

        self._socket = data_ws.FyersDataSocket(
            access_token=f"{settings.fyers_app_id}:{token}",
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self._on_connect,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )
        self._socket.connect()

    def subscribe(self, symbols: Iterable[str]) -> None:
        """Add symbols to the desired set and subscribe when connected."""
        additions = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
        if not additions:
            return

        with self._lock:
            self._symbols.update(additions)

        if self._connected and self._socket:
            self._socket.subscribe(
                symbols=sorted(additions),
                data_type=self.DATA_TYPE,
            )

    def unsubscribe(self, symbols: Iterable[str]) -> None:
        """Remove symbols from the desired set and FYERS subscription."""
        removals = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
        if not removals:
            return

        with self._lock:
            self._symbols.difference_update(removals)

        if self._connected and self._socket:
            self._socket.unsubscribe(
                symbols=sorted(removals),
                data_type=self.DATA_TYPE,
            )

    def keep_running(self) -> None:
        """Block while the FYERS SDK maintains the WebSocket connection."""
        if not self._socket:
            raise RuntimeError("WebSocket is not connected")
        self._socket.keep_running()
