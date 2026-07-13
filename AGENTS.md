# Agent Instructions

## Current phase

The submission branch keeps a reproducible experiment workflow: data
normalization, hospital/population/healthcare-supply feature construction,
score layers, candidate generation, the canonical C0-C13 live-agent condition
track (deterministic C0 baseline; live Gemini/OpenCode Go conditions C1, C4,
C9-C13; manual-harness conditions C2/C3 (vanilla, no Skills) and C5-C8
(AGENTS.md + filesystem Skills harness + branch search) across four CLIs —
Antigravity, Codex, Claude Code, OpenCode; see
`docs/experiment_design.md`), the condition-comparison judge (historically
"E13"; artifacts `reports/condition_comparison_report.md`,
`condition_judge_scores.jsonl`, `condition_judge_manifest.json`), the
candidate-level deliberation pipeline (critical reviewer findings, author
responses, and a provenance judge per candidate; see
`experiments/candidate_deliberation_runtime.py`), and the S1-S7/E14
site-selection track.

Live-agent rules: a failed live condition keeps its true failure mode
(`live_auth_failed` / `live_rate_limited` / `output_truncated` /
`waiting_for_manual_harness` / `live_error`) with
`comparable_for_e13 = false` (the record field keeps the historical name) —
never silently substitute deterministic rankings for live-agent output. Four
provider/model stacks must each keep the same provider and model on both
sides or the condition-comparison judge marks the comparison confounded:
C1/C9 (gemini), C2/C6 (codex), C3/C7 (claude_code), and the C4/C8/C10 trio
(opencode_go/deepseek-v4-flash).

Every condition produces a proposal report; the reports are decision-support
drafts. Cautionary language is consolidated in the single closing
`## Limitations / Required Due Diligence` section of each report (via
`geo_strategist.reporting.footer`) — do not scatter disclaimers through report
bodies, and do not reintroduce output-blocking gates. Preserve numeric
provenance controls and per-field evidence grades.

## Geo Strategist operating rules

- Use only real project data. Do not fabricate hospitals, population figures,
  coordinates, costs, facility facts, land parcels, travel times, revenues, or
  API responses.
- Preserve provenance for every concrete numeric or domain claim. If a value
  is unavailable, use an evidence grade such as `not_available`,
  `model_estimate`, `scenario_assumption`, or `unverified_candidate`.
- Never substitute C0 deterministic fallback rankings for a failed live-agent
  condition. Record the true failure mode.
- Generated experiment code must run only through the project sandbox or the
  approved code-execution guard.
- C2/C3 are vanilla no-Skills manual harness baselines. If running C2 or C3,
  do not execute the Skills-unified contract and do not perform branch search.
- C5/C6/C7/C8 are Skills-unified + branch-search harness conditions. If
  running C5-C8, read the filesystem Skill packages under `.agents/skills/`
  and, for Claude Code/OpenCode compatibility, `.claude/skills/`.
- For C5-C8, execute the Skills-unified contract in order and record a
  `skill_trace` row for each Skill package, including status, produced
  outputs, and artifact references.
- Save the final C5-C8 harness output to the exact `manual_result.json` path
  specified by the launcher prompt.

## Security

- Do not open, print, summarize, copy, modify, or commit the real `.env` file.
- Keep `.env`, `.venv`, `.scratch`, `.junk`, `.cache`, `.data`, `.runs`, and
  `references/local` ignored by Git.
- Do not add real secrets, credentials, raw private files, or local-only source
  material to tracked files.

## Real-data-only rule

All numeric or domain values in pipeline code, tests, configs, reports, and
committed data files must come from real sources. This rule is strictly enforced.

**Prohibited in committed files:**
- Mock datasets, fake hospitals, synthetic populations, dummy demand figures,
  invented land prices, placeholder revenues, or any hard-coded invented domain
  records
- Committed test fixtures that imitate real population, hospital, land, finance,
  or API data
- Placeholder outputs that look like real analysis results

**Permitted only as implementation mechanics:**
- Temporary test files generated at runtime under `tmp_path` with purely
  structural content (no domain claims)
- Documented scenario assumptions stored in `configs/` when no direct data
  source exists, clearly labeled as assumptions with estimation basis

See `docs/context/real_data_only_implementation_policy.md` for the full policy.

## Scope

- Actual sourced project inputs (hospital workbook facts, municipality names,
  population figures, candidate IDs) are permitted and encouraged in code,
  tests, and reports when they are traceable and evidence-status labeled.
- Do not invent, fabricate, or synthesize domain values (hospital names,
  population figures, coordinates, revenues, land prices).
- Preserve provenance (source_file_hash, locator, calculation_trace) on every
  numeric claim in every pipeline output.
- Do not create mock numeric datasets or fake API responses.

## Validation

Run all of the following checks after any change:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/validate_no_mock_data.py .
.venv/bin/python scripts/validate_normalized_outputs.py
.venv/bin/python scripts/validate_analysis_views.py
.venv/bin/python -m geo_strategist.cli validate-config
.venv/bin/python -m geo_strategist.cli validate-views
.venv/bin/python scripts/audit_prototype_safety.py .
git check-ignore .env .venv .scratch .junk .cache .data .runs references/local
git status --short
```

All commands must pass before committing. Do not add `--no-verify` or otherwise
bypass validation hooks.

## Commit hygiene

- Stage only intended source files. Never stage `.env`, `.data`, `.cache`,
  `.runs`, `.venv`, or `references/local`.
- Use explicit `git add <file>` rather than `git add .` or `git add -A`.
- Commit messages should state what changed and why, not just what was typed.
