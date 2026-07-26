"""Provider preflight: probe every live provider and harness, write status.

Implements ``python -m geo_strategist.cli check-live-agent-providers``.
Writes under ``outputs/provider_preflight/``:

- ``provider_status.json`` — one row per provider/harness (no secrets)
- ``gemini_probe.json`` / ``opencode_go_probe.json`` / ``openrouter_probe.json``
- ``harness_status.md``

Endpoint *categories* are recorded (e.g. ``chat_completions``), never URLs
that could embed credentials, and API keys are never written.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_strategist.harnesses.status import all_harness_statuses
from geo_strategist.providers.base import CallLedger, ChatResult
from geo_strategist.providers.gemini_client import GeminiClient
from geo_strategist.providers.opencode_go_client import OpenCodeGoClient
from geo_strategist.providers.openrouter_client import OpenRouterClient

PREFLIGHT_DIR = Path("outputs/provider_preflight")


@dataclass(frozen=True)
class PreflightResult:
    output_dir: Path
    provider_status: list[dict[str, Any]]
    all_primary_ok: bool


def _probe_row(result: ChatResult, *, endpoint_category: str, expected: str | None = None) -> dict[str, Any]:
    status = "ok" if result.ok else result.error_class
    if result.ok and expected is not None and expected not in result.text:
        status = "unexpected_response"
    return {
        "provider": result.provider,
        "model": result.model,
        "endpoint_category": endpoint_category,
        "status": status,
        "error_class": None if result.ok else result.error_class,
        "error_detail": result.error_detail,
        "retry_after": result.retry_after,
        "request_count": result.request_count,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "latency_seconds": result.latency_seconds,
        "response_excerpt": result.text[:200],
    }


def run_provider_preflight(repo_root: str | Path = ".") -> PreflightResult:
    root = Path(repo_root).resolve()
    out_dir = root / PREFLIGHT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    ledger = CallLedger()
    status_rows: list[dict[str, Any]] = []

    # Gemini direct (Interactions API)
    gemini = GeminiClient(ledger=ledger)
    gemini_result = gemini.generate(
        "Reply with exactly: GEMINI_PREFLIGHT_OK", purpose="preflight",
    )
    gemini_row = _probe_row(gemini_result, endpoint_category="interactions",
                            expected="GEMINI_PREFLIGHT_OK")
    _write_json(out_dir / "gemini_probe.json", {**gemini_row, "generated_at": generated_at})
    status_rows.append(gemini_row)

    # OpenCode Go direct (chat completions, deepseek-v4-flash)
    opencode_go = OpenCodeGoClient(ledger=ledger)
    opencode_result = opencode_go.chat(
        [{"role": "user", "content": "Reply with exactly: OPENCODE_GO_OK"}],
        purpose="preflight",
        max_tokens=128,
    )
    opencode_row = _probe_row(opencode_result, endpoint_category="chat_completions",
                              expected="OPENCODE_GO_OK")
    _write_json(out_dir / "opencode_go_probe.json", {**opencode_row, "generated_at": generated_at})
    status_rows.append(opencode_row)

    # OpenRouter (secondary judge only, when configured)
    openrouter_row: dict[str, Any] | None = None
    if os.environ.get("OPENROUTER_API_KEY"):
        openrouter = OpenRouterClient(ledger=ledger)
        openrouter_result = openrouter.generate(
            "Reply with exactly: OPENROUTER_OK", purpose="preflight", max_tokens=128,
        )
        openrouter_row = _probe_row(openrouter_result, endpoint_category="chat_completions",
                                    expected="OPENROUTER_OK")
        openrouter_row["role"] = "secondary_judge_or_configured_fallback_only"
        _write_json(out_dir / "openrouter_probe.json", {**openrouter_row, "generated_at": generated_at})
        status_rows.append(openrouter_row)

    # Harness CLIs
    harness_statuses = [status.to_dict() for status in all_harness_statuses()]
    for status in harness_statuses:
        status_rows.append({
            "provider": status["harness"],
            "model": status.get("model"),
            "endpoint_category": "cli_harness",
            "status": (
                "available" if status["available"] and status.get("logged_in") is not False
                else "available_login_unknown" if status["available"]
                else "unavailable"
            ),
            "error_class": None,
            "non_interactive_supported": status["non_interactive_supported"],
            "notes": status["notes"],
        })

    _write_json(out_dir / "provider_status.json", {
        "generated_at": generated_at,
        "providers": status_rows,
        "usage": ledger.summary(),
    })
    _write_harness_md(out_dir / "harness_status.md", harness_statuses, generated_at)

    primary_ok = (
        gemini_row["status"] == "ok" and opencode_row["status"] == "ok"
    )
    return PreflightResult(output_dir=out_dir, provider_status=status_rows,
                           all_primary_ok=primary_ok)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_harness_md(path: Path, statuses: list[dict[str, Any]], generated_at: str) -> None:
    lines = [
        "# Coding-Agent Harness Status",
        "",
        f"Generated: {generated_at}",
        "",
        "| harness | available | logged in | model | non-interactive |",
        "| --- | --- | --- | --- | --- |",
    ]
    for status in statuses:
        lines.append(
            f"| {status['harness']} | {status['available']} | {status.get('logged_in')} "
            f"| {status.get('model')} | {status['non_interactive_supported']} |"
        )
    lines.append("")
    for status in statuses:
        if status["notes"]:
            lines.append(f"## {status['harness']}")
            lines.append("")
            lines.extend(f"- {note}" for note in status["notes"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
