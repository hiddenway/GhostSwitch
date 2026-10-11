"""The current date and time for prompts; the model has no clock of its own."""

from __future__ import annotations

from datetime import datetime


def now_line() -> str:
    """A prompt line with the server's local date, time, and UTC offset."""
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    return (f"Current date and time: {now.strftime('%A, %d %B %Y, %H:%M')} "
            f"(UTC{offset[:3]}:{offset[3:]}). Use it for questions about today, dates, ages, or "
            "how long ago something happened; do not guess another date. Your training data may end "
            "earlier, which does not make this date wrong or in the future.")


def today_line() -> str:
    """A prompt line with the server's local date, the same all day.

    An agent resends its whole conversation on every step; a line that changes every minute
    would make the model compute everything after it again.
    """
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    return (f"Current date: {now.strftime('%A, %d %B %Y')} (UTC{offset[:3]}:{offset[3:]}). "
            "Do not guess another date. Your training data may end earlier, which does not make "
            "this date wrong or in the future.")
