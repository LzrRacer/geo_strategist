"""OpenCode Go (Zen) OpenAI-compatible chat-completions client.

Uses ``OPENCODE_API_KEY`` against ``OPENCODE_GO_CHAT_COMPLETIONS_URL``.
Both ``message.content`` and ``message.reasoning_content`` are parsed;
reasoning text is kept only for trace artifacts, never for final reports.
``finish_reason == "length"`` is classified ``output_truncated`` and retried
once with doubled ``max_tokens`` (capped). Token minimums per call class:
preflight >= 128, task >= ``OPENCODE_GO_TASK_MAX_TOKENS`` (default 4096),
report >= ``OPENCODE_GO_REPORT_MAX_TOKENS`` (default 8192).
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

DEFAULT_CHAT_URL = "https://opencode.ai/zen/go/v1/chat/completions"
PROVIDER_NAME = "opencode_go"
_TRUNCATION_TOKEN_CAP = 32768


def default_model() -> str:
    return os.environ.get("OPENCODE_GO_MODEL") or os.environ.get(
        "OPENCODE_GO_FAST_MODEL", "deepseek-v4-flash"
    )


def max_tokens_for(purpose: str) -> int:
    if purpose == "preflight":
        return max(int(os.environ.get("OPENCODE_GO_PREFLIGHT_MAX_TOKENS", "128")), 128)
    if purpose == "report":
        return max(int(os.environ.get("OPENCODE_GO_REPORT_MAX_TOKENS", "8192")), 8192)
    return max(int(os.environ.get("OPENCODE_GO_TASK_MAX_TOKENS", "4096")), 4096)


class OpenCodeGoClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        ledger: CallLedger | None = None,
        timeout: float = 600.0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.model = model or default_model()
        self.chat_url = os.environ.get("OPENCODE_GO_CHAT_COMPLETIONS_URL", DEFAULT_CHAT_URL)
        self.ledger = ledger
        self.timeout = timeout
        self.retry = retry or RetryPolicy()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        purpose: str = "task",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """One chat-completions call with truncation-aware retry."""

        key = os.environ.get("OPENCODE_API_KEY")
        use_model = model or self.model
        if not key:
            result = ChatResult(
                provider=PROVIDER_NAME, model=use_model,
                error_class="live_auth_failed",
                error_detail="OPENCODE_API_KEY not set", request_count=0,
            )
            self._record(result, purpose)
            return result

        tokens = max(max_tokens or 0, max_tokens_for(purpose))
        result = self._chat_once(messages, use_model, key, tokens, temperature)
        # Reasoning models spend max_tokens on reasoning before content, so a
        # truncated call may carry no content at all; escalate twice.
        for _retry in range(2):
            if result.error_class != "output_truncated" or tokens >= _TRUNCATION_TOKEN_CAP:
                break
            tokens = min(tokens * 2, _TRUNCATION_TOKEN_CAP)
            retried = self._chat_once(messages, use_model, key, tokens, temperature)
            retried.request_count += result.request_count
            result = retried
        self._record(result, purpose)
        return result

    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "task",
                 model: str | None = None, max_tokens: int | None = None) -> ChatResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, purpose=purpose, max_tokens=max_tokens)

    def _record(self, result: ChatResult, purpose: str) -> None:
        if self.ledger is not None:
            self.ledger.record(result, purpose=purpose)

    def _chat_once(
        self,
        messages: list[dict[str, str]],
        model: str,
        key: str,
        max_tokens: int,
        temperature: float | None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        attempts = 0
        last: ChatResult | None = None
        while attempts < self.retry.max_attempts:
            started = time.time()
            try:
                response = requests.post(
                    self.chat_url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last = ChatResult(
                    provider=PROVIDER_NAME, model=model,
                    error_class="live_error",
                    error_detail=f"request_failed: {type(exc).__name__}",
                    latency_seconds=round(time.time() - started, 3),
                    request_count=attempts + 1,
                )
                attempts += 1
                sleep_for_retry(self.retry, attempts, None)
                continue
            elapsed = round(time.time() - started, 3)
            if response.status_code != 200:
                retry_after = _retry_after_seconds(response)
                last = ChatResult(
                    provider=PROVIDER_NAME, model=model,
                    error_class=classify_http_status(response.status_code),
                    error_detail=f"http_{response.status_code}",
                    retry_after=retry_after,
                    latency_seconds=elapsed,
                    request_count=attempts + 1,
                )
                if last.error_class == "live_auth_failed":
                    return last
                attempts += 1
                sleep_for_retry(self.retry, attempts, retry_after)
                continue
            try:
                body = response.json()
            except json.JSONDecodeError:
                last = ChatResult(
                    provider=PROVIDER_NAME, model=model,
                    error_class="live_error", error_detail="invalid_json_response",
                    latency_seconds=elapsed, request_count=attempts + 1,
                )
                attempts += 1
                continue
            parsed = parse_chat_completions_response(body, model=model)
            parsed.latency_seconds = elapsed
            parsed.request_count = attempts + 1
            return parsed
        return last or ChatResult(
            provider=PROVIDER_NAME, model=model,
            error_class="live_error", error_detail="no_attempts_made", request_count=0,
        )


def parse_chat_completions_response(body: dict[str, Any], *, model: str) -> ChatResult:
    """Parse an OpenAI-compatible response: content + reasoning_content +
    usage (including nested reasoning tokens) + length-truncation class."""

    choices = body.get("choices") or []
    message = (choices[0].get("message") if choices else None) or {}
    finish_reason = choices[0].get("finish_reason") if choices else None
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    result = ChatResult(
        provider=PROVIDER_NAME,
        model=str(body.get("model") or model),
        text=str(message.get("content") or ""),
        reasoning_text=str(message.get("reasoning_content") or ""),
        finish_reason=str(finish_reason) if finish_reason else None,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0),
    )
    if finish_reason == "length":
        result.error_class = "output_truncated"
        result.error_detail = "finish_reason_length"
    elif not result.text and not result.reasoning_text:
        result.error_class = "live_error"
        result.error_detail = "empty_message"
    return result


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
