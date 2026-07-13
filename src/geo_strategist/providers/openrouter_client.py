"""OpenRouter client — secondary judge / explicitly configured fallback only.

Never a silent replacement for a failed primary provider: a 429 after
retries is reported as ``live_rate_limited`` so callers surface the failure
instead of substituting deterministic output. Retry honors the server's
``retry_after_seconds`` hint when present.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from geo_strategist.providers.base import (
    CallLedger,
    ChatResult,
    RetryPolicy,
    classify_http_status,
    sleep_for_retry,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_NAME = "openrouter"


def default_model() -> str:
    return os.environ.get("OPENROUTER_E13_MODEL", "qwen/qwen3-coder:free")


class OpenRouterClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        ledger: CallLedger | None = None,
        timeout: float = 300.0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.model = model or default_model()
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.ledger = ledger
        self.timeout = timeout
        self.retry = retry or RetryPolicy(max_attempts=4, base_delay=5.0)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "secondary_judge",
        max_tokens: int = 4096,
    ) -> ChatResult:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            result = ChatResult(
                provider=PROVIDER_NAME, model=self.model,
                error_class="live_auth_failed",
                error_detail="OPENROUTER_API_KEY not set", request_count=0,
            )
            self._record(result, purpose)
            return result
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}

        attempts = 0
        last: ChatResult | None = None
        while attempts < self.retry.max_attempts:
            started = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last = ChatResult(
                    provider=PROVIDER_NAME, model=self.model,
                    error_class="live_error",
                    error_detail=f"request_failed: {type(exc).__name__}",
                    latency_seconds=round(time.time() - started, 3),
                    request_count=attempts + 1,
                )
                attempts += 1
                sleep_for_retry(self.retry, attempts, None)
                continue
            elapsed = round(time.time() - started, 3)
            body: dict[str, Any] = {}
            try:
                body = response.json()
            except json.JSONDecodeError:
                pass
            # OpenRouter can return HTTP 200 with an embedded upstream error.
            embedded = body.get("error") if isinstance(body, dict) else None
            status = response.status_code
            if embedded and status == 200:
                status = int(embedded.get("code") or 500)
            if status != 200:
                retry_after = _retry_after_from(response, embedded)
                last = ChatResult(
                    provider=PROVIDER_NAME, model=self.model,
                    error_class=classify_http_status(status),
                    error_detail=f"http_{status}",
                    retry_after=retry_after,
                    latency_seconds=elapsed,
                    request_count=attempts + 1,
                )
                if last.error_class == "live_auth_failed":
                    break
                attempts += 1
                sleep_for_retry(self.retry, attempts, retry_after)
                continue
            choices = body.get("choices") or []
            message = (choices[0].get("message") if choices else None) or {}
            usage = body.get("usage") or {}
            result = ChatResult(
                provider=PROVIDER_NAME,
                model=str(body.get("model") or self.model),
                text=str(message.get("content") or ""),
                reasoning_text=str(message.get("reasoning_content") or message.get("reasoning") or ""),
                finish_reason=str(choices[0].get("finish_reason")) if choices else None,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                latency_seconds=elapsed,
                request_count=attempts + 1,
            )
            if not result.text:
                result.error_class = "live_error"
                result.error_detail = "empty_message"
            self._record(result, purpose)
            return result
        final = last or ChatResult(
            provider=PROVIDER_NAME, model=self.model,
            error_class="live_error", error_detail="no_attempts_made", request_count=0,
        )
        self._record(final, purpose)
        return final

    def _record(self, result: ChatResult, purpose: str) -> None:
        if self.ledger is not None:
            self.ledger.record(result, purpose=purpose)


def _retry_after_from(response: requests.Response, embedded: dict[str, Any] | None) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    if embedded:
        metadata = embedded.get("metadata") or {}
        value = metadata.get("retry_after_seconds")
        if isinstance(value, (int, float)):
            return float(value)
    return None
