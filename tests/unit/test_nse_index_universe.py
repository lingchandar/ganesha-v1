from src.ingestion.nse_index_universe import fetch_index_constituents, load_controlled_index_universe

class FakeResponse:
    def __init__(self, data): self._data = data
    def raise_for_status(self): pass
    def json(self): return {"data": self._data}

class FakeSession:
    def __init__(self, responses): self.responses = responses; self.calls = []
    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.responses[params["index"]])

def test_fetch_index_constituents_normalizes_symbols():
    session = FakeSession({"NIFTY 50": [{"symbol":"TCS"},{"symbol":"INFY"},{"symbol":"TCS"}]})
    assert fetch_index_constituents("NIFTY50", session=session) == ("NSE:INFY-EQ", "NSE:TCS-EQ")

def test_controlled_universe_deduplicates_overlap():
    session = FakeSession({"NIFTY 50": [{"symbol":"TCS"},{"symbol":"INFY"}], "NIFTY BANK": [{"symbol":"TCS"},{"symbol":"HDFCBANK"}]})
    assert load_controlled_index_universe(session=session) == ("NSE:HDFCBANK-EQ", "NSE:INFY-EQ", "NSE:TCS-EQ")

def test_unsupported_index_fails():
    try:
        fetch_index_constituents("SENSEX", session=FakeSession({}))
    except ValueError as exc:
        assert "Unsupported NSE index" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
