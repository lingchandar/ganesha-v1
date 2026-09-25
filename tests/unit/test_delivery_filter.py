"""
Unit tests for Module 7 (Institutional Delivery Percentage Filter)
and Module 16a (Expiry-Week Delivery Threshold Tightening).
"""
import pytest
from src.ingestion.nse_delivery_client import NSEArchiveDeliveryClient


def test_delivery_expansion_normal_session_pass():
    client = NSEArchiveDeliveryClient()
    # 5 days of delivery volume at 100,000 avg
    history = [100000, 110000, 90000, 105000, 95000] # SMA = 100,000
    today_delivery = 160000 # 1.6x SMA (> 1.5x)

    res = client.calculate_delivery_expansion(history, today_delivery, is_expiry_week=False)
    assert res["is_institutional_accumulation"] is True
    assert res["expansion_ratio"] == 1.60
    assert res["threshold_required"] == 1.50
    assert res["rejection_reason"] is None


def test_delivery_expansion_normal_session_fail():
    client = NSEArchiveDeliveryClient()
    history = [100000, 100000, 100000, 100000, 100000] # SMA = 100,000
    today_delivery = 140000 # 1.4x SMA (< 1.5x)

    res = client.calculate_delivery_expansion(history, today_delivery, is_expiry_week=False)
    assert res["is_institutional_accumulation"] is False
    assert res["expansion_ratio"] == 1.40
    assert "DELIVERY_EXPANSION_INSUFFICIENT" in res["rejection_reason"]


def test_delivery_expansion_expiry_week_tightening():
    client = NSEArchiveDeliveryClient()
    history = [100000, 100000, 100000, 100000, 100000] # SMA = 100,000
    
    # 1.6x passes normal threshold (1.5x), but MUST FAIL in expiry week (1.725x)
    today_delivery = 160000
    res_expiry = client.calculate_delivery_expansion(history, today_delivery, is_expiry_week=True)
    assert res_expiry["is_institutional_accumulation"] is False
    assert res_expiry["threshold_required"] == 1.725
    assert "1.60x < 1.725x" in res_expiry["rejection_reason"]

    # 1.8x passes both normal and expiry threshold
    today_delivery_surge = 180000
    res_pass = client.calculate_delivery_expansion(history, today_delivery_surge, is_expiry_week=True)
    assert res_pass["is_institutional_accumulation"] is True
    assert res_pass["expansion_ratio"] == 1.80


def test_delivery_expansion_insufficient_history():
    client = NSEArchiveDeliveryClient()
    history = [100000, 100000] # only 2 days
    res = client.calculate_delivery_expansion(history, 200000, is_expiry_week=False)
    assert res["is_institutional_accumulation"] is False
    assert "LESS_THAN_5_DAYS" in res["rejection_reason"]
