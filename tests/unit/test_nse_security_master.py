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



def test_parse_actual_nse_udiff_column_names():
    raw = (
        "FinInstrmId,TckrSymb,SctySrs,FinInstrmNm,ISIN,SctyStsNrmlMkt\n"
        "12345,RELIANCE,EQ,Reliance Industries Limited,INE002A01018,1\n"
        "12346,NDTV-RE,BE,NDTV Rights Entitlement,INE999R01010,6\n"
    ).encode()

    result = parse_security_master_csv(raw)

    assert result["symbol"].tolist() == ["RELIANCE", "NDTV-RE"]
    assert result["series"].tolist() == ["EQ", "BE"]
    assert result["instrument_token"].tolist() == [12345, 12346]
    assert result["company_name"].tolist() == [
        "Reliance Industries Limited",
        "NDTV Rights Entitlement",
    ]
    assert result["security_description"].tolist() == [
        "Reliance Industries Limited",
        "NDTV Rights Entitlement",
    ]
    assert result["status"].tolist() == ["1", "6"]


def test_equity_filter_keeps_equity_series_and_excludes_re_and_etf():
    df = pd.DataFrame(
        {
            "symbol": [
                "RELIANCE",
                "3IINFOLTD",
                "NDTV-RE",
                "NIFTYBEES",
                "AAKAAR",
                "DEBTSEC",
            ],
            "series": ["EQ", "BE", "BE", "EQ", "SM", "N0"],
            "security_description": [
                "Reliance Industries Limited",
                "3i Infotech Limited",
                "NDTV Rights Entitlement",
                "Nippon India ETF Nifty BeES",
                "Aakaar Engineering SME",
                "Non Convertible Debenture",
            ],
        }
    )

    result = filter_equity_series(df)

    assert result["symbol"].tolist() == [
        "RELIANCE",
        "3IINFOLTD",
        "AAKAAR",
    ]


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
        content = b"landing"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [{
                    "displayName": module.NSE_SECURITY_REPORT_NAME,
                    "fileActlName": "NSE_CM_security_25092026.csv.gz",
                    "filePath": "https://nsearchives.nseindia.com/content/equities/",
                }]
            }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FakeResponse()

    fake = FakeSession()
    monkeypatch.setattr(module, "_nse_session", lambda: fake)

    raw, url = module.download_security_master(date(2026, 9, 25))

    assert raw == b"landing"
    assert fake.calls[0][0] == module.NSE_REPORTS_URL
    assert fake.calls[1][0] == module.NSE_REPORTS_API_URL
    assert fake.calls[2][0] == url



def test_download_raises_when_reports_api_has_no_security_master(monkeypatch):
    from src.ingestion import nse_security_master as module

    class FakeResponse:
        content = b"landing"

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    class FakeSession:
        def get(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(module, "_nse_session", lambda: FakeSession())

    with pytest.raises(FileNotFoundError):
        module.download_security_master(date(2026, 9, 25))



def test_find_report_file_matches_requested_security_master():
    from src.ingestion.nse_security_master import _find_report_file

    payload = {
        "data": [{
            "displayName": "CM - MII - Security File (.gz) (NSE Listed securities)",
            "fileActlName": "NSE_CM_security_25092026.csv.gz",
            "filePath": "https://nsearchives.nseindia.com/content/equities/",
            "tradingDate": "25-Sep-2026",
        }]
    }

    result = _find_report_file(payload, date(2026, 9, 25))

    assert result["fileActlName"] == "NSE_CM_security_25092026.csv.gz"
    assert result["filePath"].endswith("/content/equities/")


def test_find_report_file_returns_none_for_missing_date():
    from src.ingestion.nse_security_master import _find_report_file

    payload = {
        "data": [{
            "fileActlName": "NSE_CM_security_24092026.csv.gz",
            "filePath": "https://nsearchives.nseindia.com/content/equities/",
        }]
    }

    assert _find_report_file(payload, date(2026, 9, 25)) is None


def test_download_uses_nse_reports_api(monkeypatch):
    from src.ingestion import nse_security_master as module

    class FakeResponse:
        def __init__(self, content=b"", payload=None):
            self.content = content
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if url == module.NSE_REPORTS_URL:
                return FakeResponse(content=b"landing")
            if url == module.NSE_REPORTS_API_URL:
                return FakeResponse(payload={
                    "data": [{
                        "displayName": module.NSE_SECURITY_REPORT_NAME,
                        "fileActlName": "NSE_CM_security_25092026.csv.gz",
                        "filePath": "https://nsearchives.nseindia.com/content/equities/",
                    }]
                })
            return FakeResponse(content=b"gzipped-data")

    fake = FakeSession()
    monkeypatch.setattr(module, "_nse_session", lambda: fake)

    raw, url = module.download_security_master(date(2026, 9, 25))

    assert raw == b"gzipped-data"
    assert url.endswith("NSE_CM_security_25092026.csv.gz")
    assert fake.calls[0][0] == module.NSE_REPORTS_URL
    assert fake.calls[1][0] == module.NSE_REPORTS_API_URL
    assert fake.calls[2][0] == url


def test_download_accepts_direct_gzip_from_reports_api(monkeypatch):
    from src.ingestion import nse_security_master as module

    gzip_payload = b"\x1f\x8b\x08direct-security-master"

    class FakeResponse:
        content = gzip_payload
        headers = {"Content-Type": "application/x-gzip"}
        status_code = 200
        url = module.NSE_REPORTS_API_URL
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            raise AssertionError("Direct gzip response must not be parsed as JSON")

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FakeResponse()

    fake = FakeSession()
    monkeypatch.setattr(module, "_nse_session", lambda: fake)

    raw, url = module.download_security_master(date(2026, 9, 25))

    assert raw == gzip_payload
    assert url == module.NSE_REPORTS_API_URL
    assert [call[0] for call in fake.calls] == [
        module.NSE_REPORTS_URL,
        module.NSE_REPORTS_API_URL,
    ]
