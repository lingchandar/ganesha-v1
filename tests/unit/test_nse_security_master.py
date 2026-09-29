from datetime import date
import gzip
import io
import pandas as pd
import pytest

from src.ingestion.nse_security_master import (
    build_security_master_url,
    filter_equity_series,
    parse_security_master_csv,
    security_snapshot_hash,
)


def test_builds_official_nse_security_master_url():
    assert build_security_master_url(date(2026, 9, 25)).endswith(
        "NSE_CM_security_25092026.csv.gz"
    )


def test_parses_gzipped_security_master_and_preserves_series():
    raw_csv = (
        "SYMBOL,SERIES,ISIN,NAME OF COMPANY\n"
        "AAA,EQ,INE000000001,Alpha Ltd\n"
        "AAA,BE,INE000000001,Alpha Ltd\n"
        "BBB,EQ,INE000000002,Beta Ltd\n"
    ).encode()
    raw = gzip.compress(raw_csv)

    df = parse_security_master_csv(raw)

    assert len(df) == 3
    assert set(df["series"]) == {"EQ", "BE"}
    assert df.loc[df["symbol"] == "AAA", "company_name"].iloc[0] == "Alpha Ltd"


def test_parses_pipe_delimited_master():
    raw = (
        "SYMBOL|SERIES|ISIN|NAME OF COMPANY|TOKEN\n"
        "AAA|EQ|INE000000001|Alpha Ltd|12345\n"
        "BBB|BE|INE000000002|Beta Ltd|12346\n"
    ).encode()

    df = parse_security_master_csv(raw)

    assert df["symbol"].tolist() == ["AAA", "BBB"]
    assert df["series"].tolist() == ["EQ", "BE"]
    assert int(df.loc[df["symbol"] == "AAA", "instrument_token"].iloc[0]) == 12345


def test_rejects_master_without_symbol_and_series():
    with pytest.raises(ValueError, match="missing required columns"):
        parse_security_master_csv(b"ISIN,NAME\nINE1,Alpha\n")


def test_equity_filter_is_separate_from_ingestion():
    df = pd.DataFrame(
        {"symbol": ["AAA", "AAA", "BBB"], "series": ["EQ", "BE", "EQ"]}
    )
    result = filter_equity_series(df)
    assert result["symbol"].tolist() == ["AAA", "BBB"]


def test_to_fyers_equity_symbol():
    from src.ingestion.nse_security_master import to_fyers_equity_symbol

    assert to_fyers_equity_symbol("RELIANCE") == "NSE:RELIANCE-EQ"
    assert to_fyers_equity_symbol("NSE:TCS-EQ") == "NSE:TCS-EQ"


def test_snapshot_hash_is_stable():
    assert security_snapshot_hash(b"abc") == security_snapshot_hash(b"abc")
    assert security_snapshot_hash(b"abc") != security_snapshot_hash(b"abcd")


def test_download_warms_nse_reports_session(monkeypatch):
    from src.ingestion import nse_security_master as module

    class FakeResponse:
        def __init__(self, content=b"data"):
            self.content = content

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FakeResponse()

    fake = FakeSession()
    monkeypatch.setattr(module, "_nse_session", lambda: fake)

    raw, url = module.download_security_master(date(2026, 9, 25))

    assert raw == b"data"
    assert url.endswith("NSE_CM_security_25092026.csv.gz")
    assert fake.calls[0][0] == module.NSE_REPORTS_URL
    assert fake.calls[1][0] == url
    assert fake.calls[1][1]["headers"]["Referer"] == module.NSE_REPORTS_URL


def test_download_preserves_nse_archive_error_after_session_warmup(monkeypatch):
    from src.ingestion import nse_security_master as module

    class FakeResponse:
        content = b""

        def raise_for_status(self):
            raise module.requests.HTTPError("404")

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(url)
            return FakeResponse() if url != module.NSE_REPORTS_URL else type(
                "WarmResponse", (), {"content": b"reports"}
            )()

    fake = FakeSession()
    monkeypatch.setattr(module, "_nse_session", lambda: fake)

    with pytest.raises(module.requests.HTTPError):
        module.download_security_master(date(2026, 9, 25))

    assert fake.calls == [
        module.NSE_REPORTS_URL,
        module.build_security_master_url(date(2026, 9, 25)),
    ]
