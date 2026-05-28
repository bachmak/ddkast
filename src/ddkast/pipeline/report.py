from __future__ import annotations

import smtplib
from datetime import UTC, date, datetime, timedelta
from email.message import EmailMessage

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore

_console = Console()


def _slice_tomorrow_utc(predictions: pd.Series[float]) -> pd.Series[float]:
    """Return predictions for tomorrow 00:00–23:00 UTC (24 hourly rows)."""
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    start = pd.Timestamp(tomorrow, tz="UTC")
    end = start + pd.Timedelta(hours=23)
    window: pd.Series[float] = predictions.loc[start:end]  # type: ignore[misc]
    if len(window) != 24:
        raise ValueError(
            f"Expected 24 hourly predictions for {tomorrow} UTC, got {len(window)}. "
            f"Predictions span {predictions.index[0]} → {predictions.index[-1]}. "
            "Check config.horizon (must reach tomorrow 23:00 UTC)."
        )
    return window


def _compose(window: pd.Series[float], forecast_date: date) -> tuple[str, str]:
    subject = f"ddkast forecast for {forecast_date} (UTC)"
    index: pd.DatetimeIndex = window.index  # type: ignore[assignment]
    rows = "\n".join(
        f"{ts.strftime('%H:%M')},{value:.2f}"
        for ts, value in zip(index, window.to_numpy(), strict=True)
    )
    body = f"Forecast for {forecast_date} UTC (MW):\n\n{rows}\n"
    return subject, body


def _send(subject: str, body: str, config: Config) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.email_from
    msg["To"] = config.email_to
    msg.set_content(body)

    with smtplib.SMTP(config.smtp_host, config.smtp_port) as smtp:
        if config.smtp_use_tls:
            smtp.starttls()
        if config.smtp_user:
            smtp.login(config.smtp_user, config.smtp_password.get_secret_value())
        smtp.send_message(msg)


def _require_email_config(config: Config) -> None:
    missing = [
        name
        for name, value in (
            ("SMTP_HOST", config.smtp_host),
            ("EMAIL_FROM", config.email_from),
            ("EMAIL_TO", config.email_to),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing required email configuration: {', '.join(missing)}. "
            "Set these via environment variables or .env."
        )


def run(config: Config) -> None:
    """Slice tomorrow's forecast from predict output and email it."""
    _require_email_config(config)

    processed = ParquetStore(config.processed_dir)
    predictions: pd.Series[float] = processed.read(config.processed_predictions)[
        config.model_target
    ]

    window = _slice_tomorrow_utc(predictions)
    index: pd.DatetimeIndex = window.index  # type: ignore[assignment]
    forecast_date: date = index[0].date()
    subject, body = _compose(window, forecast_date)

    _console.print(
        f"[bold]report[/bold]  sending forecast for {forecast_date} (UTC) "
        f"→ {config.email_to}…"
    )
    _send(subject, body, config)
    _console.print(f"  [green]✓[/green] email sent ({len(window)} hourly rows)")
