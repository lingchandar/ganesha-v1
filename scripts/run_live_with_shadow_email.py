"""Run the live monitor and email every shadow prediction line.

This is intentionally a terminal-output alert bridge. It does not change
candidate-model logic, EntryGate behavior, or place orders.
"""

from __future__ import annotations

import argparse
import os
import smtplib
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Sequence


PREFIX = "SHADOW ENTRY PREDICTION | "


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(env: dict[str, str], key: str, default: str = "") -> str:
    return os.getenv(key, env.get(key, default))


def email_enabled(env: dict[str, str]) -> bool:
    return setting(env, "SHADOW_EMAIL_ENABLED", "false").lower() in {
        "1", "true", "yes", "on"
    }


def send_email(env: dict[str, str], shadow_line: str) -> None:
    host = setting(env, "SHADOW_SMTP_HOST")
    port = int(setting(env, "SHADOW_SMTP_PORT", "587"))
    username = setting(env, "SHADOW_SMTP_USERNAME")
    password = setting(env, "SHADOW_SMTP_PASSWORD")
    sender = setting(env, "SHADOW_EMAIL_FROM", username)
    recipient = setting(env, "SHADOW_EMAIL_TO")
    use_tls = setting(env, "SHADOW_SMTP_USE_TLS", "true").lower() in {
        "1", "true", "yes", "on"
    }

    missing = [
        name for name, value in {
            "SHADOW_SMTP_HOST": host,
            "SHADOW_SMTP_USERNAME": username,
            "SHADOW_SMTP_PASSWORD": password,
            "SHADOW_EMAIL_FROM": sender,
            "SHADOW_EMAIL_TO": recipient,
        }.items() if not value
    ]
    if missing:
        raise RuntimeError(
            "Shadow email is enabled but missing: " + ", ".join(missing)
        )

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = shadow_line.replace(
        PREFIX, "GANESHA V1 SHADOW ENTRY | ", 1
    )
    message.set_content(
        f"{shadow_line}\n\n"
        "STATUS: CANDIDATE-ONLY SHADOW PREDICTION\n"
        "This email is an alert that the terminal prediction appeared. "
        "It is NOT a confirmed entry and does NOT place an order.\n"
    )

    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        smtp.login(username, password)
        smtp.send_message(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Ganesha live monitor with shadow-prediction email alerts"
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument(
        "--shadow-model",
        type=Path,
        default=Path("models/entry_prediction_extra_trees_candidate_v1.joblib"),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("config/.env"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = load_dotenv(args.env_file)

    if email_enabled(env):
        print(
            "SHADOW EMAIL ALERTS: ENABLED | "
            f"TO={setting(env, 'SHADOW_EMAIL_TO')}",
            flush=True,
        )
    else:
        print("SHADOW EMAIL ALERTS: DISABLED", flush=True)

    command = [
        sys.executable,
        "-m",
        "src.ingestion.live_swing_monitor",
        "--symbols",
        *args.symbols,
        "--shadow-model",
        str(args.shadow_model),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if not line.startswith(PREFIX) or not email_enabled(env):
            continue
        try:
            send_email(env, line.rstrip("\n"))
            print("SHADOW EMAIL SENT", flush=True)
        except Exception as exc:
            print(f"SHADOW EMAIL ERROR | {exc}", flush=True)

    return_code = process.wait()
    print(f"LIVE MONITOR EXIT CODE: {return_code}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
