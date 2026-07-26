"""Antigravity (`agy`) CLI support: workspace trust, model resolution, preflight.

Findings behind this module (probed against the installed `agy` 1.1.5 CLI,
2026-07-21, see `outputs/condition_proposals/live/runs/C05/diagnostics/`):

- `agy --print` (headless mode) cannot render an interactive approval
  prompt. A tool call that needs permission on a directory the CLI does not
  already consider safe is auto-*denied* (not queued, not retried) with the
  literal message: "a tool required the ... permission that headless mode
  cannot prompt for, so it was auto-denied". The CLI's own suggested
  remediations are either an allow-rule in `permissions.allow`, or
  `--dangerously-skip-permissions` (a global bypass this project must not
  use for Antigravity).
- A directory listed under `trustedWorkspaces` in
  `~/.gemini/antigravity-cli/settings.json` is exempt from that per-tool
  approval gate — this is the same mechanism this project's real repository
  root already relies on (it was trusted from prior interactive use, which
  is why earlier automated C5 runs could read/write/execute there at all).
  Registering the sanitized no-Skills workspace under the same mechanism is
  a scoped, auditable, per-directory trust decision — not a global
  permissions bypass.
- `agy models` lists effort-suffixed slugs (e.g. `gemini-3.5-flash-medium`),
  not the bare provider/model strings this project's condition registry
  uses (e.g. `gemini-3.5-flash`). `--model <slug-from-agy-models>` is
  required; passing the bare registry model string 400s.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _settings_path() -> Path:
    override = os.environ.get("GEO_STRATEGIST_ANTIGRAVITY_SETTINGS_PATH")
    if override:
        return Path(override)
    return Path.home() / ".gemini" / "antigravity-cli" / "settings.json"


@dataclass(frozen=True)
class TrustRegistration:
    workspace: str
    was_already_trusted: bool
    trusted_now: bool
    settings_path: str
    note: str = ""


def workspace_is_trusted(workspace: Path) -> bool:
    path = _settings_path()
    if not path.exists():
        return False
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    trusted = settings.get("trustedWorkspaces") or []
    return str(workspace.resolve()) in trusted


def ensure_workspace_trusted(workspace: Path) -> TrustRegistration:
    """Add ``workspace`` to agy's ``trustedWorkspaces`` if not already present.

    Scoped to exactly this one directory; every other key in the settings
    file (and every other trusted path) is preserved untouched. Never
    raises — a missing/corrupt/unwritable settings file is reported via
    ``note`` with ``trusted_now=False`` rather than crashing the harness run.
    """

    path = _settings_path()
    workspace_str = str(workspace.resolve())
    try:
        if path.exists():
            settings = json.loads(path.read_text(encoding="utf-8"))
        else:
            settings = {}
    except (OSError, json.JSONDecodeError) as exc:
        return TrustRegistration(
            workspace=workspace_str, was_already_trusted=False, trusted_now=False,
            settings_path=str(path), note=f"could not read {path}: {exc}")

    trusted = list(settings.get("trustedWorkspaces") or [])
    if workspace_str in trusted:
        return TrustRegistration(
            workspace=workspace_str, was_already_trusted=True, trusted_now=True,
            settings_path=str(path))

    trusted.append(workspace_str)
    settings["trustedWorkspaces"] = trusted
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        return TrustRegistration(
            workspace=workspace_str, was_already_trusted=False, trusted_now=False,
            settings_path=str(path), note=f"could not write {path}: {exc}")
    return TrustRegistration(
        workspace=workspace_str, was_already_trusted=False, trusted_now=True,
        settings_path=str(path), note="added to trustedWorkspaces")


# ---------------------------------------------------------------------------
# Model slug resolution: registry model string -> agy's effort-suffixed slug
# ---------------------------------------------------------------------------

_DEFAULT_EFFORT = "medium"


def _run_agy_models(timeout: float = 20.0) -> list[str]:
    binary = shutil.which("agy")
    if not binary:
        return []
    try:
        completed = subprocess.run(
            ["agy", "models"], capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def resolve_agy_model_slug(
    configured_model: str | None,
    *,
    effort: str = _DEFAULT_EFFORT,
    available_slugs: list[str] | None = None,
) -> str | None:
    """Map a registry model string (e.g. ``gemini-3.5-flash``) to the slug
    ``agy --model`` actually accepts (e.g. ``gemini-3.5-flash-medium``).

    Returns ``None`` when no confident match exists rather than guessing —
    callers should fall back to omitting ``--model`` (agy's own default)
    and record the miss as a preflight note.
    """

    if not configured_model:
        return None
    slugs = available_slugs if available_slugs is not None else _run_agy_models()
    if not slugs:
        return None
    exact = f"{configured_model}-{effort}"
    if exact in slugs:
        return exact
    if configured_model in slugs:
        return configured_model
    prefix_matches = [slug for slug in slugs if slug.startswith(f"{configured_model}-")]
    if exact in prefix_matches:
        return exact
    if prefix_matches:
        return sorted(prefix_matches)[0]
    return None


# ---------------------------------------------------------------------------
# Bounded preflight / smoke test
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    resolved_model: str | None = None
    detail: str = ""


def run_antigravity_preflight(
    repo_root: Path,
    *,
    configured_model: str | None = None,
    test_output_dir: Path | None = None,
    timeout_seconds: float = 90.0,
) -> PreflightResult:
    """Bounded smoke test that a full C5 run is actually possible.

    Verifies (in order, short-circuiting on the first hard failure):
    1. the `agy` binary is on PATH;
    2. `agy --version` succeeds;
    3. `agy models` returns a parseable model list and the configured model
       resolves to a real slug;
    4. a trivial `agy --print` round trip succeeds (checks auth + the
       non-interactive path itself, not just the binary);
    5. the process can read a real repository file (`AGENTS.md` or
       `pyproject.toml`) from ``repo_root``;
    6. the process can create (and this preflight cleans up) a small file in
       an approved test output directory.

    Never raises. A step it cannot verify is recorded as a note, not a
    silent pass — this is a smoke test, not a full run, so it intentionally
    does not exercise tool-approval behavior (see `antigravity_support`
    module docstring for that finding, established separately against the
    installed CLI).
    """

    checks: dict[str, bool] = {}
    notes: list[str] = []

    binary = shutil.which("agy")
    checks["binary_on_path"] = binary is not None
    if not binary:
        return PreflightResult(ok=False, checks=checks, notes=["agy not found on PATH"],
                                detail="binary_missing")

    try:
        version = subprocess.run(["agy", "--version"], capture_output=True, text=True,
                                  timeout=20.0, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        checks["version_check"] = False
        return PreflightResult(ok=False, checks=checks, notes=[f"agy --version failed: {exc}"],
                                detail="version_check_failed")
    checks["version_check"] = version.returncode == 0
    if version.returncode != 0:
        notes.append(f"agy --version exited {version.returncode}: {version.stderr.strip()[:300]}")
        return PreflightResult(ok=False, checks=checks, notes=notes, detail="version_check_failed")

    slugs = _run_agy_models()
    checks["model_list_available"] = bool(slugs)
    resolved_model = resolve_agy_model_slug(configured_model, available_slugs=slugs) if slugs else None
    checks["configured_model_resolves"] = resolved_model is not None
    if not slugs:
        notes.append("agy models returned no output; cannot confirm --model will resolve")
    elif resolved_model is None:
        notes.append(
            f"configured model {configured_model!r} did not resolve against agy models output "
            f"({slugs[:8]}); a run without --model would use agy's own default, which may not "
            "match the strict-pair contract")

    try:
        probe = subprocess.run(
            ["agy", "--print", "Reply with exactly the single word OK and nothing else. "
                                "Do not use any tools.",
             *(["--model", resolved_model] if resolved_model else []),
             "--print-timeout", "60s"],
            capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        checks["print_roundtrip"] = False
        notes.append(f"agy --print smoke prompt exceeded the {timeout_seconds}s preflight timeout")
        return PreflightResult(ok=False, checks=checks, notes=notes, resolved_model=resolved_model,
                                detail="print_roundtrip_timeout")
    except OSError as exc:
        checks["print_roundtrip"] = False
        return PreflightResult(ok=False, checks=checks, notes=[f"agy --print failed to launch: {exc}"],
                                resolved_model=resolved_model, detail="print_roundtrip_launch_failed")
    lowered = (probe.stdout + probe.stderr).lower()
    auth_markers = ("not logged in", "authentication", "please run agy install",
                    "no credentials", "unauthorized")
    if any(marker in lowered for marker in auth_markers):
        checks["print_roundtrip"] = False
        notes.append(f"agy --print smoke prompt looks like an auth failure: {lowered[:300]}")
        return PreflightResult(ok=False, checks=checks, notes=notes, resolved_model=resolved_model,
                                detail="auth_failure")
    checks["print_roundtrip"] = probe.returncode == 0
    if probe.returncode != 0:
        notes.append(f"agy --print smoke prompt exited {probe.returncode}: "
                      f"{(probe.stdout + probe.stderr).strip()[:300]}")

    agents_md = repo_root / "AGENTS.md"
    pyproject = repo_root / "pyproject.toml"
    readable = agents_md if agents_md.exists() else pyproject
    checks["repo_file_readable"] = readable.exists()
    if not readable.exists():
        notes.append(f"neither {agents_md} nor {pyproject} exists to confirm repository reads")

    test_dir = test_output_dir or (repo_root / ".scratch" / "antigravity_preflight")
    write_ok = False
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        probe_file = test_dir / "preflight_write_probe.txt"
        probe_file.write_text("preflight\n", encoding="utf-8")
        write_ok = probe_file.exists()
        probe_file.unlink(missing_ok=True)
        if not any(test_dir.iterdir()):
            test_dir.rmdir()
    except OSError as exc:
        notes.append(f"could not create/clean a test file under {test_dir}: {exc}")
    checks["can_write_test_output_dir"] = write_ok

    hard_requirements = ("binary_on_path", "version_check", "print_roundtrip", "repo_file_readable",
                          "can_write_test_output_dir")
    ok = all(checks.get(name, False) for name in hard_requirements)
    return PreflightResult(ok=ok, checks=checks, notes=notes, resolved_model=resolved_model,
                            detail="ok" if ok else "one or more preflight checks failed")
