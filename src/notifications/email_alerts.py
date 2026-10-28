"""SMTP email alerts for read-only live Ganesha signals.

This adapter is intentionally notification-only. It never opens, modifies, or
cancels an order and it does not promote the candidate ML model to production.
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any


@dataclass(frozen=True)
class EmailAlertConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str
    enabled: bool

    @classmethod
    def from_environment(cls) -> "EmailAlertConfig":
        enabled = os.getenv("GANESHA_EMAIL_ENABLED", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        return cls(
            host=os.getenv("GANESHA_EMAIL_SMTP_HOST", "smtp.gmail.com"),
            port=int(os.getenv("GANESHA_EMAIL_SMTP_PORT", "587")),
            username=os.getenv("GANESHA_EMAIL_USERNAME", ""),
            password=os.getenv("GANESHA_EMAIL_PASSWORD", ""),
            sender=os.getenv("GANESHA_EMAIL_FROM", ""),
            recipient=os.getenv("GANESHA_EMAIL_TO", ""),
            enabled=enabled,
        )

    @property
    def ready(self) -> bool:
        return bool(
            self.enabled
            and self.host
            and self.port
            and self.username
            and self.password
            and self.sender
            and self.recipient
        )


class EmailAlertSender:
    """Send one-shot SMTP alerts; failures are surfaced to the caller."""

    def __init__(self, config: EmailAlertConfig | None = None) -> None:
        self.config = config or EmailAlertConfig.from_environment()

    @property
    def enabled(self) -> bool:
        return self.config.ready

    def send_entry_candidate(self, signal: Any, shadow: Any) -> None:
        """Send a next-session candidate alert after a completed daily signal."""
        if not self.enabled:
            return

        probability = float(shadow.probability_target_before_stop)
        confidence = float(shadow.confidence)
        subject = (
            "GANESHA V1 | ENTRY CANDIDATE | "
            f"{signal.ticker_symbol} | {signal.setup_type}"
        )
        body = "\n".join(
            [
                "GANESHA V1 — READ-ONLY ENTRY CANDIDATE ALERT",
                "",
                f"Symbol: {signal.ticker_symbol}",
                f"Setup: {signal.setup_type}",
                f"Reference entry: {float(signal.entry_price):.2f}",
                f"Stop loss: {float(signal.stop_loss):.2f}",
                f"Target: {float(signal.target_price):.2f}",
                f"Risk/Reward: {float(signal.risk_reward_ratio):.2f}R",
                f"Shadow probability: {probability:.4f}",
                f"Shadow confidence: {confidence:.4f}",
                f"Model: {shadow.model_version}",
                "",
                "STATUS: CANDIDATE ONLY — NOT A PRODUCTION ENTRY.",
                "No order has been placed or requested.",
                "The validated live engine evaluates the completed daily session,",
                "so this alert is for next-session review rather than an intraday buy command.",
                "",
                f"Rationale: {signal.rationale}",
            ]
        )

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.config.sender
        message["To"] = self.config.recipient
        message.set_content(body)

        with smtplib.SMTP(self.config.host, self.config.port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(self.config.username, self.config.password)
            smtp.send_message(message)
