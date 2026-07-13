"""Shared provider-call result and usage-accounting types.

``ChatResult`` normalizes every provider response into text +
reasoning + usage + an ``error_class`` drawn from the execution-mode
vocabulary used across the condition track:

- ``ok`` — live call succeeded
- ``live_auth_failed`` — 401/403 or missing key
- ``live_rate_limited`` — 429 (``retry_after`` carries the server hint)
- ``output_truncated`` — finish_reason == "length" after all retries
- ``live_error`` — any other failure (network, 5xx, parse)

``CallLedger`` accumulates per-model request/token counts so conditions can
emit model-call summary tables without re-parsing traces. API keys are never
stored on any of these objects.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResult:
    provider: str
    model: str
    text: str = ""
    reasoning_text: str = ""
    finish_reason: str | None = None
    error_class: str = "ok"
    error_detail: str | None = None
    retry_after: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_seconds: float = 0.0
    request_count: int = 1

    @property
    def ok(self) -> bool:
        return self.error_class == "ok"


def classify_http_status(status: int) -> str:
    if status in (401, 403):
        return "live_auth_failed"
    if status == 429:
        return "live_rate_limited"
    return "live_error"


class CallLedger:
    """Thread-safe per-(purpose, model) usage accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str, str], dict[str, Any]] = {}

    def record(self, result: ChatResult, *, purpose: str = "task") -> None:
        key = (result.provider, result.model, purpose)
        with self._lock:
            row = self._rows.setdefault(key, {
                "provider": result.provider,
                "model": result.model,
                "purpose": purpose,
                "request_count": 0,
                "error_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_latency_seconds": 0.0,
            })
            row["request_count"] += result.request_count
            row["error_count"] += 0 if result.ok else 1
            row["prompt_tokens"] += result.prompt_tokens
            row["completion_tokens"] += result.completion_tokens
            row["reasoning_tokens"] += result.reasoning_tokens
            row["total_latency_seconds"] = round(
                row["total_latency_seconds"] + result.latency_seconds, 3
            )

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for _key, row in sorted(self._rows.items())]

    def summary(self) -> dict[str, Any]:
        rows = self.rows()
        return {
            "total_requests": sum(r["request_count"] for r in rows),
            "total_errors": sum(r["error_count"] for r in rows),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
            "total_reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
            "by_model": rows,
        }

    def requests_by_model(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows():
            counts[row["model"]] = counts.get(row["model"], 0) + row["request_count"]
        return counts


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 65.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after + 1.0, self.max_delay)
        return min(self.base_delay * (2 ** attempt), self.max_delay)


def sleep_for_retry(policy: RetryPolicy, attempt: int, retry_after: float | None) -> None:
    time.sleep(policy.delay(attempt, retry_after))


_REDACT_MARKERS = ("KEY", "TOKEN", "SECRET", "AUTHORIZATION", "PASSWORD")


def redact_secrets(text: str, env: dict[str, str] | None = None) -> str:
    """Best-effort removal of secret values from free text before persisting."""

    import os

    source = env if env is not None else dict(os.environ)
    for name, value in source.items():
        if not value or len(value) < 8:
            continue
        if any(marker in name.upper() for marker in _REDACT_MARKERS):
            text = text.replace(value, f"<redacted:{name}>")
    return text
