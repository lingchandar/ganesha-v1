"""
NSE CM Security Master ingestion.

Downloads the official NSE CM-MII daily security master, validates its
structure, normalizes common NSE column-name variants, and writes the raw
daily universe snapshot. Eligibility is intentionally kept separate from
ingestion so the project can ingest the full NSE universe first.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import text
import requests


NSE_SECURITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/"
    "NSE_CM_security_{date:%d%m%Y}.csv.gz"
)

COLUMN_ALIASES = {
    "symbol": {"SYMBOL", "SYMBOL_NAME", "TICKER"},
    "series": {"SERIES", "SERIES_CODE"},
    "isin": {"ISIN", "ISIN_NUMBER"},
    "company_name": {"NAME_OF_COMPANY", "COMPANY_NAME", "NAME"},
    "status": {"STATUS", "SECURITY_STATUS"},
    "instrument_token": {"INSTRUMENT_TOKEN", "TOKEN"},
}


@dataclass(frozen=True)
class SecurityRecord:
    symbol: str
    series: str
    isin: str | None
    company_name: str | None
    status: str | None
    instrument_token: int | None


def build_security_master_url(snapshot_date: date) -> str:
    return NSE_SECURITY_URL.format(date=snapshot_date)


def _normalise_columns(columns: Iterable[str]) -> dict[str, str]:
    normalised = {str(c).strip().upper().replace(" ", "_"): str(c) for c in columns}
    mapping: dict[str, str] = {}
    for logical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised:
                mapping[logical] = normalised[alias]
                break
    return mapping


def _read_delimited_payload(payload: bytes) -> pd.DataFrame:
    """Read NSE security-master payloads with comma/pipe delimiter detection."""
    sample = payload[:65536].decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|;\\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "|" if "|" in sample.splitlines()[0] else ","
    return pd.read_csv(
        io.BytesIO(payload),
        dtype=str,
        sep=delimiter,
        engine="python",
    )


def parse_security_master_csv(raw: bytes) -> pd.DataFrame:
    """Parse gzip-compressed or plain NSE security-master bytes."""
    payload = raw
    if raw[:2] == b"\x1f\x8b":
        payload = gzip.decompress(raw)

    df = _read_delimited_payload(payload)
    mapping = _normalise_columns(df.columns)
    missing = {"symbol", "series"} - mapping.keys()
    if missing:
        raise ValueError(
            f"NSE security master missing required columns: {sorted(missing)}; "
            f"received={list(df.columns)}"
        )

    rename = {source: logical for logical, source in mapping.items()}
    df = df.rename(columns=rename)

    for column in ("isin", "company_name", "status", "instrument_token"):
        if column not in df.columns:
            df[column] = None

    for column in ("symbol", "series"):
        df[column] = df[column].fillna("").astype(str).str.strip().str.upper()

    df = df[df["symbol"] != ""].copy()
    df = df.drop_duplicates(subset=["symbol", "series"], keep="last")

    if df.empty:
        raise ValueError("NSE security master contains no securities")

    if df["symbol"].duplicated().any():
        # A symbol can legitimately have multiple series; keep that distinction.
        pass

    if "instrument_token" in df.columns:
        df["instrument_token"] = pd.to_numeric(
            df["instrument_token"], errors="coerce"
        ).astype("Int64")

    for column in ("isin", "company_name", "status"):
        df[column] = df[column].where(df[column].notna(), None)

    return df


def to_fyers_equity_symbol(symbol: str) -> str:
    """Convert a raw NSE equity symbol to FYERS equity-symbol format."""
    value = str(symbol).strip().upper()
    if value.startswith("NSE:"):
        return value if value.endswith("-EQ") else f"{value}-EQ"
    return f"NSE:{value}-EQ"


def persist_current_security_master(
    db,
    snapshot_date: date,
    df: pd.DataFrame,
    *,
    source: str,
    source_file: str,
    source_sha256: str,
) -> int:
    """Replace the current raw NSE CM master while retaining first/last seen dates."""
    required = {"symbol", "series", "isin", "company_name", "status", "instrument_token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Security master frame missing columns: {sorted(missing)}")

    rows = df.to_dict("records")
    for row in rows:
        db.execute(
            text("""
                INSERT INTO nse_security_master_current
                    (symbol, series_code, isin, company_name, security_status,
                     nse_security_token, first_seen_date, last_seen_date,
                     source, source_file, source_sha256)
                VALUES
                    (:symbol, :series, :isin, :company_name, :status,
                     :token, :snapshot_date, :snapshot_date,
                     :source, :source_file, :source_sha256)
                ON CONFLICT (symbol, series_code)
                DO UPDATE SET
                    isin = EXCLUDED.isin,
                    company_name = EXCLUDED.company_name,
                    security_status = EXCLUDED.security_status,
                    nse_security_token = EXCLUDED.nse_security_token,
                    last_seen_date = EXCLUDED.last_seen_date,
                    source = EXCLUDED.source,
                    source_file = EXCLUDED.source_file,
                    source_sha256 = EXCLUDED.source_sha256,
                    updated_at = CURRENT_TIMESTAMP
            """),
            {
                "symbol": row["symbol"],
                "series": row["series"],
                "isin": row.get("isin"),
                "company_name": row.get("company_name"),
                "status": row.get("status"),
                "token": int(row["instrument_token"]) if pd.notna(row["instrument_token"]) else None,
                "snapshot_date": snapshot_date,
                "source": source,
                "source_file": source_file,
                "source_sha256": source_sha256,
            },
        )
    return len(rows)


def ingest_daily_security_master(
    snapshot_date: date,
    db,
    *,
    raw_directory: str | Path = "data/nse_security_master",
) -> int:
    """Download, hash, archive, parse and persist one official NSE CM master."""
    raw, source_file = download_security_master(snapshot_date)
    source_sha256 = security_snapshot_hash(raw)
    save_raw_snapshot(raw, snapshot_date, raw_directory)
    parsed = parse_security_master_csv(raw)
    persist_current_security_master(
        db,
        snapshot_date,
        parsed,
        source="NSE_CM_MII_SECURITY",
        source_file=source_file,
        source_sha256=source_sha256,
    )
    persist_security_snapshot(
        db,
        snapshot_date,
        parsed,
        source="NSE_CM_MII_SECURITY",
        source_file=source_file,
        source_sha256=source_sha256,
        universe_name="NSE_LISTED_CM",
    )
    return len(parsed)


def download_security_master(snapshot_date: date, timeout: int = 30) -> tuple[bytes, str]:
    url = build_security_master_url(snapshot_date)
    headers = {
        "User-Agent": "Mozilla/5.0 GaneshaV1/1.0",
        "Accept": "text/csv,application/gzip,application/octet-stream,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"NSE returned an empty security master: {url}")
    return response.content, url


def security_snapshot_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def filter_equity_series(
    df: pd.DataFrame,
    allowed_series: tuple[str, ...] = ("EQ",),
) -> pd.DataFrame:
    """Return equity-series rows; no liquidity/swing filter is applied here."""
    allowed = {value.upper() for value in allowed_series}
    return df[df["series"].isin(allowed)].copy()


def save_raw_snapshot(raw: bytes, snapshot_date: date, directory: str | Path) -> Path:
    path = Path(directory) / f"NSE_CM_security_{snapshot_date:%Y%m%d}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def persist_security_snapshot(
    db,
    snapshot_date: date,
    df: pd.DataFrame,
    *,
    source: str,
    source_file: str,
    source_sha256: str,
    universe_name: str = "NSE_LISTED_EQUITY",
) -> int:
    """Persist one complete daily security snapshot without applying swing filters."""
    required = {"symbol", "series", "company_name", "instrument_token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshot frame missing columns: {sorted(missing)}")

    rows = df.to_dict("records")
    for row in rows:
        db.execute(
            text("""
                INSERT INTO universe_membership_snapshot
                    (snapshot_date, universe_name, ticker_symbol, instrument_token,
                     series_code, company_name, source, source_file, source_sha256,
                     is_eligible)
                VALUES
                    (:snapshot_date, :universe_name, :ticker_symbol, :instrument_token,
                     :series_code, :company_name, :source, :source_file, :source_sha256,
                     TRUE)
                ON CONFLICT (snapshot_date, universe_name, ticker_symbol)
                DO UPDATE SET
                    instrument_token = EXCLUDED.instrument_token,
                    series_code = EXCLUDED.series_code,
                    company_name = EXCLUDED.company_name,
                    source = EXCLUDED.source,
                    source_file = EXCLUDED.source_file,
                    source_sha256 = EXCLUDED.source_sha256
            """),
            {
                "snapshot_date": snapshot_date,
                "universe_name": universe_name,
                "ticker_symbol": row["symbol"],
                "instrument_token": (
                    int(row["instrument_token"])
                    if pd.notna(row["instrument_token"]) else None
                ),
                "series_code": row["series"],
                "company_name": row.get("company_name"),
                "source": source,
                "source_file": source_file,
                "source_sha256": source_sha256,
            },
        )
    return len(rows)
