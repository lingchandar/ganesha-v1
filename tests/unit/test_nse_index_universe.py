from src.ingestion.nse_index_universe import fetch_index_constituents, load_controlled_index_universe


class FakeResponse:
    def __init__(self, csv_text):
        self.content = csv_text.encode("utf-8")

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return FakeResponse(self.responses[url])


def test_fetch_index_constituents_normalizes_symbols():
    session = FakeSession(
        {
            "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv":
                "Company Name,Symbol,Industry\nTata Consultancy Services,TCS,IT\nInfosys,INFY,IT\nTata Consultancy Services,TCS,IT\n"
        }
    )
    assert fetch_index_constituents("NIFTY50", session=session) == (
        "NSE:INFY-EQ",
        "NSE:TCS-EQ",
    )


def test_controlled_universe_deduplicates_overlap():
    session = FakeSession(
        {
            "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv":
                "Company Name,Symbol,Industry\nTCS,TCS,IT\nInfosys,INFY,IT\n",
            "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv":
                "Company Name,Symbol,Industry\nTCS,TCS,IT\nHDFC Bank,HDFCBANK,Banks\n",
        }
    )
    assert load_controlled_index_universe(session=session) == (
        "NSE:HDFCBANK-EQ",
        "NSE:INFY-EQ",
        "NSE:TCS-EQ",
    )


def test_unsupported_index_fails():
    try:
        fetch_index_constituents("SENSEX", session=FakeSession({}))
    except ValueError as exc:
        assert "Unsupported NSE index" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
