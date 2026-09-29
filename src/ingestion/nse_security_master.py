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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
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


def parse_security_master_csv(raw: bytes) -> pd.DataFrame:
    """Parse gzip-compressed or plain CSV security-master bytes."""
    payload = raw
    if raw[:2] == b"\x1f\x8b":
        payload = gzip.decompress(raw)

    df = pd.read_csv(io.BytesIO(payload), dtype=str)
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

    return df


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
