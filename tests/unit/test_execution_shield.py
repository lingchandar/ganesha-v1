"""
GANESHA V1 — Unit Tests for the SEBI Anti-Loop Execution Shield

These tests verify the foundational compliance layer. ALL tests must pass
before any other code in the system is considered trustworthy.

Run with: python -m pytest tests/unit/test_execution_shield.py -v
"""
import time
import pytest

from src.ingestion.execution_shield import (
    intercept_outbound_request,
    SEBIComplianceBreachError,
    AntiLoopThrottleGuard,
    BLOCKED_ORDER_ENDPOINTS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Order Endpoint Blocking
# Every known order-routing URL pattern must be caught and blocked.
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderEndpointBlocking:
    """Verify that ALL order-routing endpoints are blocked."""

    def test_blocks_generic_orders(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/api/v3/orders")

    def test_blocks_place_order_underscore(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/place_order")

    def test_blocks_place_order_hyphen(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/place-order")

    def test_blocks_modify_order_underscore(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/modify_order")

    def test_blocks_modify_order_hyphen(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/modify-order")

    def test_blocks_cancel_order_underscore(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/cancel_order")

    def test_blocks_cancel_order_hyphen(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/cancel-order")

    def test_blocks_fyers_v3_orders(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/api/v3/orders")

    def test_blocks_fyers_v2_orders(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/api/v2/orders")

    def test_blocks_angel_order_place(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request(
                "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder"
            )

    def test_blocks_angel_order_modify(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request(
                "https://apiconnect.angelone.in/order/modify"
            )

    def test_blocks_angel_order_cancel(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request(
                "https://apiconnect.angelone.in/order/cancel"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: GTT & Basket Order Blocking
# ═══════════════════════════════════════════════════════════════════════════════

class TestGTTAndBasketBlocking:
    """Verify GTT and basket/bulk order endpoints are blocked."""

    def test_blocks_gtt_place(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/gtt/place")

    def test_blocks_gtt_modify(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/gtt/modify")

    def test_blocks_gtt_cancel(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/gtt/cancel")

    def test_blocks_gtt_create(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/gtt/create")

    def test_blocks_gtt_delete(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/gtt/delete")

    def test_blocks_basket_order(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/basket/order")

    def test_blocks_basket_place(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/basket/place")

    def test_blocks_multi_order(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.broker.com/multi-order")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Case Insensitivity
# URLs must be caught regardless of casing.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaseInsensitivity:
    """Verify blocking works regardless of URL casing."""

    def test_uppercase_orders(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/API/V3/ORDERS")

    def test_mixed_case_place_order(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/Place_Order")

    def test_mixed_case_gtt(self):
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request("https://api.fyers.in/GTT/Place")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Allowed Endpoints (Must NOT Be Blocked)
# Market data, quotes, symbols, and history endpoints must flow through freely.
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllowedEndpoints:
    """Verify that read-only data endpoints are NOT blocked."""

    def test_allows_historical_data(self):
        # Must NOT raise any exception
        intercept_outbound_request("https://api.fyers.in/data/history")

    def test_allows_quotes(self):
        intercept_outbound_request("https://api.fyers.in/api/v3/quotes")

    def test_allows_symbols(self):
        intercept_outbound_request("https://api.fyers.in/api/v3/symbols")

    def test_allows_angel_market_quote(self):
        intercept_outbound_request(
            "https://apiconnect.angelone.in/rest/secure/angelbroking/market/v1/quote"
        )

    def test_allows_angel_historical(self):
        intercept_outbound_request(
            "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData"
        )

    def test_allows_angel_login(self):
        intercept_outbound_request(
            "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByPassword"
        )

    def test_allows_nse_website(self):
        intercept_outbound_request("https://www.nseindia.com/api/equity-stockIndices")

    def test_allows_market_depth(self):
        intercept_outbound_request("https://api.fyers.in/api/v3/depth")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: Error Message Quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorMessageQuality:
    """Verify error messages contain useful diagnostic information."""

    def test_error_contains_url(self):
        url = "https://api.fyers.in/api/v3/orders"
        with pytest.raises(SEBIComplianceBreachError, match="api.fyers.in/api/v3/orders"):
            intercept_outbound_request(url)

    def test_error_contains_compliance_tag(self):
        with pytest.raises(SEBIComplianceBreachError, match="SEBI_COMPLIANCE_BREACH"):
            intercept_outbound_request("https://api.fyers.in/place_order")

    def test_error_mentions_algo_id(self):
        with pytest.raises(SEBIComplianceBreachError, match="Algo ID"):
            intercept_outbound_request("https://api.fyers.in/orders")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Throttle Guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestThrottleGuard:
    """Verify the Anti-Loop Throttle Guard rate limiting behavior."""

    def test_enforces_minimum_interval(self):
        """Two consecutive requests should be at least 250ms apart."""
        guard = AntiLoopThrottleGuard(max_rpm=120, min_interval_ms=250)

        start = time.monotonic()
        guard.wait_for_slot()
        guard.wait_for_slot()
        elapsed = time.monotonic() - start

        # Second call should have waited at least 250ms
        assert elapsed >= 0.24, (
            f"Throttle guard failed: elapsed={elapsed:.3f}s < 0.25s"
        )

    def test_stats_tracking(self):
        """Verify that the guard tracks request counts."""
        guard = AntiLoopThrottleGuard(max_rpm=120, min_interval_ms=50)

        guard.wait_for_slot()
        guard.wait_for_slot()
        guard.wait_for_slot()

        stats = guard.get_stats()
        assert stats["requests_in_last_minute"] == 3
        assert stats["max_rpm"] == 120
        assert stats["min_interval_ms"] == 50

    def test_initialization_parameters(self):
        """Verify custom initialization parameters are stored."""
        guard = AntiLoopThrottleGuard(max_rpm=60, min_interval_ms=500)
        stats = guard.get_stats()

        assert stats["max_rpm"] == 60
        assert stats["min_interval_ms"] == 500


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7: Completeness Check
# Ensure we haven't accidentally left any blocked endpoints untested.
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockedEndpointCompleteness:
    """Verify every entry in BLOCKED_ORDER_ENDPOINTS actually triggers blocking."""

    @pytest.mark.parametrize("endpoint", BLOCKED_ORDER_ENDPOINTS)
    def test_every_blocked_endpoint_raises(self, endpoint):
        test_url = f"https://api.testbroker.com{endpoint}"
        with pytest.raises(SEBIComplianceBreachError):
            intercept_outbound_request(test_url)
