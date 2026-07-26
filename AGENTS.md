# Agent Instructions

## Current phase

The submission branch keeps a reproducible experiment workflow: data
normalization, hospital/population/healthcare-supply feature construction,
score layers, candidate generation, the canonical C0-C14 live-agent condition
track (deterministic C0 baseline; live Gemini/OpenCode Go conditions C1, C4,
C13-C14; manual-harness conditions C2/C3 (vanilla, no Skills, single pass),
C5-C8 (open-ended native coding-agent controls — no project Skills), and
C9-C12 (AGENTS.md + dynamic filesystem Skills operator library)
across four CLIs — Antigravity, Codex, Claude Code, OpenCode; see
`docs/experiment_design.md`), the condition-comparison judge (historically
"E13"; artifacts `reports/condition_comparison_report.md`,
`condition_judge_scores.jsonl`, `condition_judge_manifest.json`,
`condition_deterministic_checklist.jsonl`, `condition_llm_checklist.jsonl`),
the candidate-level deliberation pipeline (critical reviewer findings, author
responses, and a provenance judge per candidate; see
`experiments/candidate_deliberation_runtime.py`), and the S1-S7/E14
site-selection track.

C5-C8 and C9-C12 form four strict pairs isolating the project-specific
Skills contract on an otherwise identical provider/model/harness stack:
C5/C9 (Antigravity, gemini-3.5-flash), C6/C10 (Codex, gpt-5.5), C7/C11
(Claude Code, sonnet-5.0), C8/C12 (OpenCode, deepseek-v4-flash).

Live-agent rules: a failed live condition keeps its true failure mode
(`live_auth_failed` / `live_rate_limited` / `output_truncated` /
`waiting_for_manual_harness` / `live_error`) with
`comparable_for_e13 = false` (the record field keeps the historical name) —
never silently substitute deterministic rankings for live-agent output. Four
provider/model stacks must each keep the same provider and model on both
sides or the condition-comparison judge marks the comparison confounded:
C1/C13 (gemini), C2/C10 (codex), C3/C11 (claude_code), and the C4/C12/C14 trio
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
- C2/C3 are vanilla no-Skills manual harness baselines (single pass, no tool
  use). If running C2 or C3, do not execute the Skills-unified contract and
  do not perform branch search.
- C5/C6/C7/C8 are full native coding-agent controls (multi-step repository
  inspection, editing, code execution, debugging permitted) but must NOT
  install, expose, read, or otherwise use project Skills. If running C5-C8,
  do not open, summarize, or follow `.agents/skills/`, `.claude/skills/`, or
  any Skills-unified procedure. They may independently generate hypotheses,
  decision regimes, code, robustness tests, reviews, revisions, and portfolios.
  The five shared objectives are minimum coverage anchors, not a forbidden or
  fixed branch pipeline. Reading or referencing the Skill packages contaminates
  the comparison and requires a re-run.
- C9/C10/C11/C12 are Skills-unified + branch-search harness conditions. If
  running C9-C12, read the filesystem Skill packages under `.agents/skills/`
  and, for Claude Code/OpenCode compatibility, `.claude/skills/`.
- For C9-C12, select Skills dynamically and reuse them when evidence gaps,
  failed execution, new regimes, critique, or robustness results require it.
  Record every selected invocation in `skill_trace`, including status,
  produced outputs, and resolvable artifact references. The lifecycle must
  remain traceable from inspection and hypotheses through execution, review,
  revision, portfolio synthesis, and the final proposal; not every registered
  Skill must run exactly once.
- Save the final C5-C12 harness output to the exact `manual_result.json` path
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
