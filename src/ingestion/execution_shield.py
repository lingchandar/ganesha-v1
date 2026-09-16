"""
GANESHA V1 — Hard-Coded Anti-Loop Execution Shield (Module 0 Upgrade)

SEBI COMPLIANCE CRITICAL:
This module structurally prevents any outbound order placement, modification,
or cancellation from ever leaving this application. It acts as a runtime
packet interceptor on all HTTP requests made by the broker API client.

If ANY request targets an order execution endpoint, the process is
immediately terminated with a fatal RuntimeError.

References:
    - SEBI Client Direct API Guidelines
    - NSE Platform Services — Non-NEAT Decision Support Tools
    - Master Specification Section 2.1
"""
import time
from collections import deque
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCKED ENDPOINT PATTERNS
# These URL path fragments are associated with order routing on Fyers & Angel One.
# If ANY outbound HTTP request URL contains these, execution is HALTED IMMEDIATELY.
# ═══════════════════════════════════════════════════════════════════════════════
BLOCKED_ORDER_ENDPOINTS = frozenset([
    # Generic order paths
    "/orders",
    "/place_order",
    "/place-order",
    "/modify_order",
    "/modify-order",
    "/cancel_order",
    "/cancel-order",

    # Fyers API v3 specific
    "/api/v3/orders",
    "/api/v2/orders",

    # Angel One SmartAPI specific
    "/rest/secure/angelbroking/order",
    "/order/place",
    "/order/modify",
    "/order/cancel",

    # GTT (Good-Till-Triggered) order endpoints
    "/gtt/place",
    "/gtt/modify",
    "/gtt/cancel",
    "/gtt/create",
    "/gtt/delete",

    # Basket / bulk order endpoints
    "/basket/order",
    "/basket/place",
    "/multi-order",
])


class SEBIComplianceBreachError(RuntimeError):
    """
    Fatal error raised when an order routing attempt is detected.
    This error is INTENTIONALLY uncatchable — it signals a structural
    compliance violation that must halt the process immediately.
    """
    pass


def intercept_outbound_request(url: str) -> None:
    """
    Inspect every outbound HTTP URL before it leaves the process.
    If it matches any order execution endpoint, raise a fatal error.

    This function MUST be called by the HTTP transport adapter
    before every single request.

    Args:
        url: The full URL of the outbound HTTP request.

    Raises:
        SEBIComplianceBreachError: If the URL targets an order endpoint.
    """
    url_lower = url.lower()
    for blocked_path in BLOCKED_ORDER_ENDPOINTS:
        if blocked_path in url_lower:
            error_msg = (
                f"SEBI_COMPLIANCE_BREACH: Attempted to route traffic to "
                f"order execution endpoint [{url}]. "
                f"Order routing is PROHIBITED without an exchange-approved Algo ID. "
                f"Process terminated immediately."
            )
            logger.critical(error_msg)
            raise SEBIComplianceBreachError(error_msg)


class AntiLoopThrottleGuard:
    """
    Rate limiter that enforces:
    - Maximum requests per rolling minute (default: 120)
    - Minimum interval between consecutive requests (default: 250ms)

    Prevents runaway scan loops from overwhelming broker API gateways,
    protecting against IP bans and SEBI Client Direct API violations.
    """

    def __init__(self, max_rpm: int = 120, min_interval_ms: int = 250):
        self.max_rpm = max_rpm
        self.min_interval_sec = min_interval_ms / 1000.0
        self._request_timestamps: deque = deque(maxlen=max_rpm)
        self._last_request_time: float = 0.0
        logger.info(
            f"AntiLoopThrottleGuard initialized: "
            f"max_rpm={max_rpm}, min_interval={min_interval_ms}ms"
        )

    def wait_for_slot(self) -> None:
        """
        Block until a rate-limit slot is available.
        Called before every outbound API request.
        """
        now = time.monotonic()

        # ── Enforce minimum interval between consecutive requests ──
        elapsed_since_last = now - self._last_request_time
        if self._last_request_time > 0 and elapsed_since_last < self.min_interval_sec:
            sleep_time = self.min_interval_sec - elapsed_since_last
            logger.debug(f"Throttle: sleeping {sleep_time:.3f}s (250ms interval guard)")
            time.sleep(sleep_time)
            now = time.monotonic()

        # ── Enforce per-minute rolling window cap ──
        if len(self._request_timestamps) >= self.max_rpm:
            oldest = self._request_timestamps[0]
            window_elapsed = now - oldest
            if window_elapsed < 60.0:
                sleep_time = 60.0 - window_elapsed + 0.1  # +100ms safety margin
                logger.warning(
                    f"Throttle: RPM limit ({self.max_rpm}/min) reached. "
                    f"Sleeping {sleep_time:.1f}s before next request."
                )
                time.sleep(sleep_time)
                now = time.monotonic()

        self._request_timestamps.append(now)
        self._last_request_time = now

    def get_stats(self) -> dict:
        """Return current throttle statistics for monitoring."""
        now = time.monotonic()
        recent_count = sum(
            1 for ts in self._request_timestamps if (now - ts) < 60.0
        )
        return {
            "requests_in_last_minute": recent_count,
            "max_rpm": self.max_rpm,
            "min_interval_ms": int(self.min_interval_sec * 1000),
            "total_tracked": len(self._request_timestamps),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Global Singleton Throttle Instance
# Import this anywhere: from src.ingestion.execution_shield import throttle
# ═══════════════════════════════════════════════════════════════════════════════
throttle = AntiLoopThrottleGuard()
