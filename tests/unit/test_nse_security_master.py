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


def test_rejects_master_without_symbol_and_series():
    with pytest.raises(ValueError, match="missing required columns"):
        parse_security_master_csv(b"ISIN,NAME\nINE1,Alpha\n")


def test_equity_filter_is_separate_from_ingestion():
    df = pd.DataFrame(
        {"symbol": ["AAA", "AAA", "BBB"], "series": ["EQ", "BE", "EQ"]}
    )
    result = filter_equity_series(df)
    assert result["symbol"].tolist() == ["AAA", "BBB"]


def test_snapshot_hash_is_stable():
    assert security_snapshot_hash(b"abc") == security_snapshot_hash(b"abc")
    assert security_snapshot_hash(b"abc") != security_snapshot_hash(b"abcd")
