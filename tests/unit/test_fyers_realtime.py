from datetime import timezone
from unittest.mock import Mock

import pytest

from src.ingestion.fyers_realtime import FyersRealtimeClient


def test_normalize_symbol_update():
    message = {
        "symbol": "NSE:TCS-EQ",
        "ltp": 2032.4,
        "open_price": 2065.0,
        "high_price": 2085.1,
        "low_price": 2032.4,
        "prev_close_price": 2070.7,
        "vol_traded_today": 4437783,
        "last_traded_time": 1790659800,
        "type": "sf",
    }

    tick = FyersRealtimeClient.normalize_message(message)

    assert tick.symbol == "NSE:TCS-EQ"
    assert tick.ltp == pytest.approx(2032.4)
    assert tick.open_price == pytest.approx(2065.0)
    assert tick.high_price == pytest.approx(2085.1)
    assert tick.low_price == pytest.approx(2032.4)
    assert tick.previous_close == pytest.approx(2070.7)
    assert tick.volume_traded == 4437783
    assert tick.timestamp.tzinfo == timezone.utc
    assert tick.raw_message == message


def test_normalize_requires_symbol():
    with pytest.raises(ValueError, match="no symbol"):
        FyersRealtimeClient.normalize_message({"ltp": 100})


def test_subscribe_and_unsubscribe_update_state():
    client = FyersRealtimeClient(access_token="token")
    socket = Mock()
    client._socket = socket
    client._connected = True

    client.subscribe(["NSE:TCS-EQ", "NSE:INFY-EQ"])
    assert client.symbols == ("NSE:INFY-EQ", "NSE:TCS-EQ")
    socket.subscribe.assert_called_once_with(
        symbols=["NSE:INFY-EQ", "NSE:TCS-EQ"],
        data_type="SymbolUpdate",
    )

    client.unsubscribe(["NSE:INFY-EQ"])
    assert client.symbols == ("NSE:TCS-EQ",)
    socket.unsubscribe.assert_called_once_with(
        symbols=["NSE:INFY-EQ"],
        data_type="SymbolUpdate",
    )


def test_on_connect_resubscribes_desired_symbols():
    client = FyersRealtimeClient(access_token="token")
    socket = Mock()
    client._socket = socket
    client._symbols = {"NSE:TCS-EQ", "NSE:INFY-EQ"}

    client._on_connect()

    assert client.connected is True
    socket.subscribe.assert_called_once_with(
        symbols=["NSE:INFY-EQ", "NSE:TCS-EQ"],
        data_type="SymbolUpdate",
    )


def test_message_callback_receives_normalized_tick():
    handler = Mock()
    client = FyersRealtimeClient(access_token="token", tick_handler=handler)

    client._on_message(
        {
            "symbol": "NSE:TCS-EQ",
            "ltp": 2000,
            "vol_traded_today": 1000,
        }
    )

    tick = handler.call_args.args[0]
    assert tick.symbol == "NSE:TCS-EQ"
    assert tick.ltp == 2000.0
    assert tick.volume_traded == 1000


def test_connection_requires_token(monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.fyers_realtime.FyersAuthManager.load_cached_token",
        classmethod(lambda cls: None),
    )
    client = FyersRealtimeClient()

    with pytest.raises(RuntimeError, match="No valid FYERS access token"):
        client.connect()
