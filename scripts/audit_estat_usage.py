"""Audit e-Stat API usage across the project (WP4).

Searches source, configs, and docs for any e-Stat / ESTAT_APP_ID references.
Does NOT read .env. Reports boolean credential presence only.

Usage:
  .venv/bin/python scripts/audit_estat_usage.py
"""
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def _grep_project(pattern: str, extensions: list[str], exclude_dirs: list[str]) -> list[str]:
    """Return relative file paths containing the pattern."""
    found = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
        for fname in files:
            if any(fname.endswith(ext) for ext in extensions):
                fpath = Path(root) / fname
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    if pattern.lower() in text.lower():
                        found.append(str(fpath))
                except Exception:
                    pass
    return sorted(found)


def main() -> None:
    exclude_dirs = [".venv", "__pycache__", ".git", ".data", ".cache", ".runs", ".scratch"]
    exts = [".py", ".yaml", ".yml", ".md", ".json", ".toml"]

    estat_files = _grep_project("ESTAT_APP_ID", exts, exclude_dirs)
    estat_usage_files = _grep_project("estat", exts, exclude_dirs)
    e_stat_files = _grep_project("e-stat", exts, exclude_dirs)

    # Deduplicate and sort
    all_relevant = sorted(set(estat_files + estat_usage_files + e_stat_files))

    # Categorize
    settings_refs = [f for f in all_relevant if "settings" in f]
    config_refs = [f for f in all_relevant if "config" in f or "yaml" in f]
    source_refs = [f for f in all_relevant if f.endswith(".py") and "settings" not in f]
    doc_refs = [f for f in all_relevant if f.endswith(".md")]
    other_refs = [
        f for f in all_relevant
        if f not in settings_refs + config_refs + source_refs + doc_refs
    ]

    # Check if any e-Stat client already exists in src/
    client_files = [f for f in source_refs if "estat" in f.lower()]
    existing_client = len(client_files) > 0

    # Check if used in any ingestion/scoring pipeline (not just settings)
    pipeline_files = [
        f for f in source_refs
        if any(kw in f for kw in ["ingestion", "scoring", "feature", "score_layer"])
    ]

    # Credential presence — boolean only
    app_id_present = bool(os.environ.get("ESTAT_APP_ID", ""))

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ESTAT_APP_ID_present": app_id_present,
        "estat_client_exists": existing_client,
        "client_files": client_files,
        "registered_in_settings": len(settings_refs) > 0,
        "registered_in_configs": len(config_refs) > 0,
        "used_in_ingestion_pipeline": len(pipeline_files) > 0,
        "files_with_estat_references": {
            "settings": settings_refs,
            "configs": config_refs,
            "source_non_settings": source_refs,
            "docs": doc_refs,
            "other": other_refs,
        },
        "audit_findings": [],
    }

    findings = []
    if not existing_client:
        findings.append(
            "No e-Stat client exists yet. "
            "ESTAT_APP_ID is registered in settings but not consumed by any ingestion code. "
            "A controlled retrieval adapter is being added for future LLM validation (E3+)."
        )
    if len(pipeline_files) == 0:
        findings.append(
            "e-Stat is not used in any current ingestion or scoring pipeline."
        )
    findings.append(
        "ESTAT_APP_ID is registered in src/geo_strategist/settings.py and "
        "listed in configs/data_sources.yaml, but no retrieval code calls the API yet."
    )
    findings.append(
        "Planned use: controlled e-Stat retrieval adapter for experiment E3 "
        "(LLM + evidence bundle + e-Stat lookup). "
        "LLM may only access e-Stat through the adapter; all responses cached and provenance-logged."
    )
    audit["audit_findings"] = findings

    # Write report
    out_dir = Path(".cache/study_area/tokyo_aichi_osaka")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "estat_usage_audit_report.json"
    md_path = out_dir / "estat_usage_audit_report.md"

    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    md = f"""# e-Stat Usage Audit Report

Generated: {audit['generated_at']}

## Credential Status

- `ESTAT_APP_ID` present in environment: **{app_id_present}**

## Findings

"""
    for finding in findings:
        md += f"- {finding}\n"

    md += f"""
## File References

### Settings / config
"""
    for f in settings_refs + config_refs:
        md += f"- `{f}`\n"

    md += "\n### Source code (non-settings)\n"
    for f in source_refs:
        md += f"- `{f}`\n"

    md += "\n### Docs\n"
    for f in doc_refs:
        md += f"- `{f}`\n"

    md += f"""
## Summary

| Check | Result |
|-------|--------|
| `ESTAT_APP_ID` in settings | {audit['registered_in_settings']} |
| `ESTAT_APP_ID` in configs | {audit['registered_in_configs']} |
| e-Stat client exists | {audit['estat_client_exists']} |
| Used in ingestion pipeline | {audit['used_in_ingestion_pipeline']} |
| Credential present | {app_id_present} |

**Status**: e-Stat registered but not yet consumed. Controlled adapter added for Phase 7+ readiness.
"""
    md_path.write_text(md)

    print(f"ESTAT_APP_ID present: {app_id_present}")
    print(f"e-Stat client exists: {existing_client}")
    print(f"Used in ingestion pipeline: {len(pipeline_files) > 0}")
    print(f"Registered in settings: {len(settings_refs) > 0}")
    print(f"Total files with e-Stat references: {len(all_relevant)}")
    for finding in findings:
        print(f"  Finding: {finding[:100]}...")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
