#!/usr/bin/env python3
"""Phase 11 — targeted e-Stat getStatsList metadata search.

Searches e-Stat for tables matching specified Japanese keywords and caches results.
Writes output under .runs/experiments/estat_metadata_search/<run_id>/.

Usage:
    # Cache-only (no network):
    .venv/bin/python scripts/run_estat_metadata_search.py \\
        --queries 医療施設調査 医療施設 病院 一般診療所

    # Live e-Stat fetch (source .env first):
    set -a; . ./.env; set +a
    .venv/bin/python scripts/run_estat_metadata_search.py \\
        --allow-network \\
        --queries 医療施設調査 医療施設 病院 一般診療所

    # Auto-derive queries from latest E3 requests:
    .venv/bin/python scripts/run_estat_metadata_search.py \\
        --allow-network \\
        --latest-e3

Security:
    - ESTAT_APP_ID is read from environment only; never printed.
    - Credential presence is reported as boolean only.
    - Raw response bodies are not printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

E3_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "e3_estat_retrieval_llm"
SEARCH_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_metadata_search"
ESTAT_CACHE_ROOT = REPO_ROOT / ".data" / "api_raw" / "estat"

DEFAULT_QUERIES = ["医療施設調査", "医療施設", "病院", "一般診療所"]
DEFAULT_LIMIT = "10"


def _find_latest_e3() -> Path | None:
    if not E3_RUN_ROOT.exists():
        return None
    runs = [p for p in E3_RUN_ROOT.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if runs else None


def _load_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _extract_tables(data: dict) -> list[dict]:
    """Extract TABLE_INF entries from a getStatsList response."""
    tbl = (
        data.get("GET_STATS_LIST", {})
            .get("DATALIST_INF", {})
            .get("TABLE_INF", [])
    )
    if isinstance(tbl, dict):
        tbl = [tbl]
    return [t for t in tbl if isinstance(t, dict)]


def _table_summary(tbl: dict) -> dict:
    """Summarize a TABLE_INF entry — no raw blobs, no secrets."""
    def _str(x: object) -> str:
        if isinstance(x, dict):
            return x.get("$", "")
        return str(x) if x is not None else ""

    return {
        "stats_data_id": tbl.get("@id", ""),
        "stat_name": _str(tbl.get("STAT_NAME")),
        "gov_org": _str(tbl.get("GOV_ORG")),
        "statistics_name": tbl.get("STATISTICS_NAME", ""),
        "title": _str(tbl.get("TITLE")),
        "survey_date": str(tbl.get("SURVEY_DATE")) if tbl.get("SURVEY_DATE") is not None else None,
        "overall_total_number": tbl.get("OVERALL_TOTAL_NUMBER"),
    }


def _derive_queries_from_e3(e3_dir: Path) -> list[str]:
    """Extract unique keywords from E3 retrieval request query strings."""
    rr = _load_json(e3_dir / "retrieval_requests.json") or {}
    requests = rr.get("valid_requests", [])
    keywords: list[str] = []
    for req in requests:
        query = req.get("query", "")
        for token in query.split():
            if len(token) >= 2 and token not in keywords:
                keywords.append(token)
    # Always include the core survey name
    for core in DEFAULT_QUERIES:
        if core not in keywords:
            keywords.append(core)
    return keywords[:10]  # cap at 10 to avoid excessive calls


def run_metadata_search(
    queries: list[str],
    allow_network: bool = False,
    limit: str = DEFAULT_LIMIT,
    cache_root: Path = ESTAT_CACHE_ROOT,
) -> Path:
    from geo_strategist.data.estat_retrieval import (
        fetch_estat, _cache_key, _cache_path,
    )

    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = SEARCH_RUN_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    app_id_present = bool(os.environ.get("ESTAT_APP_ID", ""))
    print(f"ESTAT_APP_ID present: {str(app_id_present).lower()}")
    print(f"Allow network:        {allow_network}")
    print(f"Queries ({len(queries)}):    {', '.join(queries)}")

    search_results: list[dict] = []
    all_issues: list[dict] = []
    total_tables = 0
    total_live = 0
    total_cache = 0

    for query in queries:
        params = {"searchWord": query, "limit": limit}
        ck = _cache_key(params)
        cp = str(_cache_path(cache_root, ck, "getStatsList"))

        data, cache_hit, issue = fetch_estat(
            endpoint="getStatsList",
            params=params,
            allow_network=allow_network,
            cache_only=not allow_network,
            cache_root=cache_root,
        )

        tables: list[dict] = []
        status: str
        if data is not None:
            tables = _extract_tables(data)
            status = "ok"
            if cache_hit:
                total_cache += 1
            else:
                total_live += 1
        else:
            if issue is not None:
                code = issue.issue_code
                status = (
                    "no_credential" if code == "source_credentials_unavailable"
                    else "network_disabled" if code == "network_not_enabled"
                    else "no_data"
                )
                all_issues.append({
                    "issue_id": str(uuid.uuid4()),
                    "severity": issue.severity,
                    "issue_code": issue.issue_code,
                    "message": issue.detail,
                    "query": query,
                })
            else:
                status = "no_data"

        table_summaries = [_table_summary(t) for t in tables]
        total_tables += len(tables)

        result = {
            "query": query,
            "status": status,
            "cache_hit": cache_hit,
            "live_fetch": not cache_hit and data is not None,
            "table_count": len(tables),
            "tables": table_summaries,
            "cache_path": cp,
            "issues": [],
        }
        if issue:
            result["issues"] = [{
                "issue_code": issue.issue_code,
                "severity": issue.severity,
                "message": issue.detail,
            }]

        search_results.append(result)
        status_label = status
        if cache_hit:
            status_label += " (cache)"
        elif data is not None:
            status_label += " (live)"
        print(f"  [{status_label:25s}] {query}: {len(tables)} table(s)")

    _write_jsonl(out_dir / "metadata_search_results.jsonl", search_results)
    _write_jsonl(out_dir / "metadata_search_issues.jsonl", all_issues)

    candidate_ids = list({
        t["stats_data_id"]
        for r in search_results
        for t in r["tables"]
        if t.get("stats_data_id")
    })

    manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "queries": queries,
        "allow_network": allow_network,
        "estat_app_id_present": app_id_present,
        "query_count": len(queries),
        "live_fetch_count": total_live,
        "cache_hit_count": total_cache,
        "total_tables_found": total_tables,
        "unique_candidate_ids": candidate_ids,
        "unique_candidate_count": len(candidate_ids),
        "issue_count": len(all_issues),
        "disclaimers": [
            "Experimental metadata search only. No statistical values retrieved.",
            "No LLM proposals generated.",
            "No cash-flow, parcel selection, or final recommendations.",
        ],
    }
    _write_json(out_dir / "metadata_search_manifest.json", manifest)

    _write_report(out_dir, manifest, search_results, all_issues)

    print(f"\nTotal tables found:   {total_tables}")
    print(f"Unique candidate IDs: {len(candidate_ids)}")
    print(f"Output: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")

    return out_dir


def _write_report(out_dir: Path, manifest: dict, results: list[dict], issues: list[dict]) -> None:
    md = [
        "# e-Stat Metadata Search Report",
        "",
        f"**Run ID:** `{manifest['run_id']}`",
        f"**Generated:** {manifest['generated_at']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Queries | {manifest['query_count']} |",
        f"| Live fetches | {manifest['live_fetch_count']} |",
        f"| Cache hits | {manifest['cache_hit_count']} |",
        f"| Total tables found | {manifest['total_tables_found']} |",
        f"| Unique candidate table IDs | {manifest['unique_candidate_count']} |",
        f"| Issues | {manifest['issue_count']} |",
        "",
    ]

    if manifest["unique_candidate_ids"]:
        md += [
            "## Candidate Table IDs",
            "",
            "| statsDataId | Stat Name | Title | Survey Date |",
            "|-------------|-----------|-------|-------------|",
        ]
        seen: set[str] = set()
        for r in results:
            for t in r["tables"]:
                sid = t.get("stats_data_id", "")
                if sid and sid not in seen:
                    seen.add(sid)
                    title = t.get("title", "")[:60]
                    stat = t.get("stat_name", "")
                    sd = t.get("survey_date", "")
                    md.append(f"| `{sid}` | {stat} | {title} | {sd} |")
        md.append("")

    md += ["## Results by Query", ""]
    for r in results:
        md += [
            f"### `{r['query']}`",
            "",
            f"- Status: `{r['status']}`",
            f"- Cache hit: {r['cache_hit']}",
            f"- Live fetch: {r['live_fetch']}",
            f"- Tables found: {r['table_count']}",
            "",
        ]
        if r["tables"]:
            md += ["| statsDataId | Stat Name | Title |", "|-------------|-----------|-------|"]
            for t in r["tables"][:5]:
                sid = t.get("stats_data_id", "")
                stat = t.get("stat_name", "")
                title = (t.get("title") or t.get("statistics_name", ""))[:50]
                md.append(f"| `{sid}` | {stat} | {title} |")
            if len(r["tables"]) > 5:
                md.append(f"| ... | ({len(r['tables']) - 5} more) | |")
            md.append("")

    if issues:
        md += ["## Issues", ""]
        for iss in issues:
            md.append(f"- **[{iss.get('severity')}]** `{iss.get('issue_code')}` ({iss.get('query')}): {iss.get('message', '')}")
        md.append("")

    md += ["## Disclaimers", ""]
    for d in manifest.get("disclaimers", []):
        md.append(f"- {d}")
    md.append("")

    (out_dir / "metadata_search_report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "metadata_search_report.json").write_text(
        json.dumps({
            "run_id": manifest["run_id"],
            "query_count": manifest["query_count"],
            "live_fetch_count": manifest["live_fetch_count"],
            "cache_hit_count": manifest["cache_hit_count"],
            "total_tables_found": manifest["total_tables_found"],
            "unique_candidate_count": manifest["unique_candidate_count"],
            "issue_count": manifest["issue_count"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="e-Stat getStatsList metadata search.")
    parser.add_argument(
        "--queries", nargs="+", default=None,
        help="Japanese search keywords (default: 医療施設調査 医療施設 病院 一般診療所)",
    )
    parser.add_argument(
        "--latest-e3", action="store_true",
        help="Derive queries from latest E3 retrieval requests",
    )
    parser.add_argument("--allow-network", action="store_true", default=False)
    parser.add_argument("--limit", default=DEFAULT_LIMIT, help="Result limit per query (default 10)")
    args = parser.parse_args()

    if args.latest_e3:
        e3_dir = _find_latest_e3()
        if e3_dir is None:
            print("ERROR: No E3 run found.", file=sys.stderr)
            sys.exit(1)
        queries = _derive_queries_from_e3(e3_dir)
    elif args.queries:
        queries = args.queries
    else:
        queries = DEFAULT_QUERIES

    run_metadata_search(
        queries=queries,
        allow_network=args.allow_network,
        limit=args.limit,
        cache_root=ESTAT_CACHE_ROOT,
    )


if __name__ == "__main__":
    main()
