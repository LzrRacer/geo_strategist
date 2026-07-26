"""Gemini direct-API client (Interactions API).

Uses ``GEMINI_API_KEY`` (falling back to ``GEMINI_API_KEY2``,
``GEMINI_API_KEY3``, and ``GOOGLE_API_KEY``), endpoint base ``GEMINI_API_BASE``, and model
``GEMINI_MODEL``. In ``interactions`` mode
(``GEMINI_API_MODE``, the default) requests POST ``{base}/interactions`` and
the response is parsed from ``steps``: thought steps contribute token counts
only, and the ``model_output`` step's ``content[].text`` parts become the
result text. Raw keys are read from the environment per request and never
stored on the client or in results.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from geo_strategist.providers.base import (
    CallLedger,
    ChatResult,
    RetryPolicy,
    classify_http_status,
    sleep_for_retry,
)

DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
PROVIDER_NAME = "gemini"
_DEFAULT_RPM_BUDGET = 9
_DEFAULT_TPM_BUDGET = 220_000
_DEFAULT_RPD_BUDGET = 18
_DEFAULT_RATE_SLEEP_CAP_SECONDS = 65.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _configured_api_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value and value not in seen:
            keys.append((name, value))
            seen.add(value)
    return keys


@dataclass
class _GeminiKeyQuota:
    minute_window_start: float = 0.0
    minute_requests: int = 0
    minute_tokens: int = 0
    day: date | None = None
    day_requests: int = 0
    disabled_until: float = 0.0
    daily_exhausted: bool = False


_QUOTA_LOCK = threading.Lock()
_KEY_QUOTAS: dict[str, _GeminiKeyQuota] = {}


def _estimate_tokens(payload: dict[str, Any]) -> int:
    prompt = str(payload.get("input") or "")
    system = str(payload.get("system_instruction") or "")
    return max(1, (len(prompt) + len(system)) // 4)


def _prepare_key_for_request(label: str, estimated_tokens: int) -> bool:
    """Apply conservative per-process Gemini quota guardrails.

    The free-tier quota is account-wide, so this cannot know calls made in
    other tools. It still prevents this process from repeatedly hammering the
    same key within the minute/day budget and makes multi-key fallback orderly.
    """

    now = time.time()
    today = date.today()
    rpm = max(1, _env_int("GEMINI_REQUESTS_PER_MINUTE", _DEFAULT_RPM_BUDGET))
    tpm = max(1, _env_int("GEMINI_TOKENS_PER_MINUTE", _DEFAULT_TPM_BUDGET))
    rpd = max(1, _env_int("GEMINI_REQUESTS_PER_DAY", _DEFAULT_RPD_BUDGET))
    sleep_cap = max(0.0, _env_float(
        "GEMINI_RATE_LIMIT_SLEEP_CAP_SECONDS", _DEFAULT_RATE_SLEEP_CAP_SECONDS))
    with _QUOTA_LOCK:
        quota = _KEY_QUOTAS.setdefault(label, _GeminiKeyQuota())
        if quota.day != today:
            quota.day = today
            quota.day_requests = 0
            quota.daily_exhausted = False
        if quota.daily_exhausted or quota.day_requests >= rpd:
            return False
        if quota.disabled_until and quota.disabled_until > now:
            return False
        if now - quota.minute_window_start >= 60.0:
            quota.minute_window_start = now
            quota.minute_requests = 0
            quota.minute_tokens = 0
        wait_seconds = 0.0
        if quota.minute_requests >= rpm or quota.minute_tokens + estimated_tokens > tpm:
            wait_seconds = max(0.0, 60.0 - (now - quota.minute_window_start))
    if wait_seconds > 0.0:
        time.sleep(min(wait_seconds + 1.0, sleep_cap))
    return True


def _record_key_request(label: str, token_count: int) -> None:
    now = time.time()
    today = date.today()
    with _QUOTA_LOCK:
        quota = _KEY_QUOTAS.setdefault(label, _GeminiKeyQuota())
        if quota.day != today:
            quota.day = today
            quota.day_requests = 0
            quota.daily_exhausted = False
        if now - quota.minute_window_start >= 60.0:
            quota.minute_window_start = now
            quota.minute_requests = 0
            quota.minute_tokens = 0
        quota.minute_requests += 1
        quota.minute_tokens += max(1, token_count)
        quota.day_requests += 1


def _mark_key_rate_limited(label: str, response: requests.Response,
                           retry_after: float | None) -> None:
    body = getattr(response, "text", "") or ""
    lowered = body.lower()
    daily = (
        "perday" in lowered
        or "per day" in lowered
        or "requestsperday" in lowered
        or "rateday" in lowered
    )
    now = time.time()
    with _QUOTA_LOCK:
        quota = _KEY_QUOTAS.setdefault(label, _GeminiKeyQuota())
        if daily:
            quota.daily_exhausted = True
            return
        delay = retry_after if retry_after is not None else 60.0
        quota.disabled_until = max(quota.disabled_until, now + min(delay + 1.0, 3600.0))


def default_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


class GeminiClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        ledger: CallLedger | None = None,
        timeout: float = 300.0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.model = model or default_model()
        self.api_base = os.environ.get("GEMINI_API_BASE", DEFAULT_API_BASE).rstrip("/")
        self.mode = os.environ.get("GEMINI_API_MODE", "interactions")
        self.ledger = ledger
        self.timeout = timeout
        self.retry = retry or RetryPolicy(
            max_attempts=_env_int("GEMINI_RETRY_ATTEMPTS", 1),
            base_delay=_env_float("GEMINI_RETRY_BASE_DELAY", 2.0),
            max_delay=_env_float("GEMINI_RETRY_MAX_DELAY", 65.0),
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "task",
    ) -> ChatResult:
        keys = _configured_api_keys()
        if not keys:
            result = ChatResult(
                provider=PROVIDER_NAME, model=self.model,
                error_class="live_auth_failed",
                error_detail="GEMINI_API_KEY/GEMINI_API_KEY2/GEMINI_API_KEY3/GOOGLE_API_KEY not set",
                request_count=0,
            )
            self._record(result, purpose)
            return result

        payload: dict[str, Any] = {"model": self.model, "input": prompt}
        if system:
            payload["system_instruction"] = system
        result = self._post_interactions(payload, keys)
        self._record(result, purpose)
        return result

    def _record(self, result: ChatResult, purpose: str) -> None:
        if self.ledger is not None:
            self.ledger.record(result, purpose=purpose)

    def _post_interactions(self, payload: dict[str, Any],
                           keys: list[tuple[str, str]]) -> ChatResult:
        url = f"{self.api_base}/interactions"
        cycles = 0
        request_count = 0
        last: ChatResult | None = None
        estimated_tokens = _estimate_tokens(payload)
        while cycles < self.retry.max_attempts:
            attempted_key = False
            for label, key in keys:
                if not _prepare_key_for_request(label, estimated_tokens):
                    continue
                attempted_key = True
                _record_key_request(label, estimated_tokens)
                request_count += 1
                started = time.time()
                try:
                    response = requests.post(
                        url,
                        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:
                    last = ChatResult(
                        provider=PROVIDER_NAME, model=self.model,
                        error_class="live_error",
                        error_detail=f"request_failed: {type(exc).__name__}",
                        latency_seconds=round(time.time() - started, 3),
                        request_count=request_count,
                    )
                    continue
                elapsed = round(time.time() - started, 3)
                if response.status_code != 200:
                    retry_after = _retry_after_seconds(response)
                    error_class = classify_http_status(response.status_code)
                    last = ChatResult(
                        provider=PROVIDER_NAME, model=self.model,
                        error_class=error_class,
                        error_detail=(
                            "http_429:all_configured_gemini_keys_rate_limited"
                            if response.status_code == 429 and len(keys) > 1 else
                            f"http_{response.status_code}"
                        ),
                        retry_after=retry_after,
                        latency_seconds=elapsed,
                        request_count=request_count,
                    )
                    if error_class == "live_rate_limited":
                        _mark_key_rate_limited(label, response, retry_after)
                        # Try the next configured key once before returning 429.
                        if len(keys) > 1:
                            continue
                        return last
                    if error_class == "live_auth_failed":
                        continue
                    continue
                try:
                    body = response.json()
                except json.JSONDecodeError:
                    last = ChatResult(
                        provider=PROVIDER_NAME, model=self.model,
                        error_class="live_error", error_detail="invalid_json_response",
                        latency_seconds=elapsed, request_count=request_count,
                    )
                    continue
                parsed = parse_interaction_response(body)
                parsed.latency_seconds = elapsed
                parsed.request_count = request_count
                return parsed
            cycles += 1
            if not attempted_key:
                last = ChatResult(
                    provider=PROVIDER_NAME, model=self.model,
                    error_class="live_rate_limited",
                    error_detail="local_gemini_quota_guard_blocked_all_configured_keys",
                    request_count=request_count,
                )
                break
            if cycles < self.retry.max_attempts:
                sleep_for_retry(self.retry, cycles, last.retry_after if last else None)
        return last or ChatResult(
            provider=PROVIDER_NAME, model=self.model,
            error_class="live_error", error_detail="no_attempts_made", request_count=0,
        )


def parse_interaction_response(body: dict[str, Any]) -> ChatResult:
    """Extract model_output text, thought presence, and usage from an
    Interactions API response body."""

    model = str(body.get("model") or default_model())
    texts: list[str] = []
    thought_steps = 0
    for step in body.get("steps") or []:
        step_type = step.get("type")
        if step_type == "thought":
            thought_steps += 1
            continue
        if step_type == "model_output":
            for part in step.get("content") or []:
                if part.get("type") == "text" and part.get("text"):
                    texts.append(str(part["text"]))
    usage = body.get("usage") or {}
    status = body.get("status")
    result = ChatResult(
        provider=PROVIDER_NAME,
        model=model,
        text="\n".join(texts),
        reasoning_text=f"<{thought_steps} thought step(s), signatures not stored>" if thought_steps else "",
        finish_reason=str(status) if status else None,
        prompt_tokens=int(usage.get("total_input_tokens") or 0),
        completion_tokens=int(usage.get("total_output_tokens") or 0),
        reasoning_tokens=int(usage.get("total_thought_tokens") or 0),
    )
    if status not in (None, "completed"):
        result.error_class = "live_error"
        result.error_detail = f"interaction_status_{status}"
    elif not result.text:
        result.error_class = "live_error"
        result.error_detail = "empty_model_output"
    return result


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
