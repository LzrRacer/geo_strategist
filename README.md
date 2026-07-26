# Geo Strategist (submission copy)

Geo Strategist is an evidence-graded experiment system for evaluating agentic
hospital-location / hospital-reorganization / candidate-site-selection
proposal generation across 15 conditions (C0–C14) over a real local evidence
base for Tokyo, Aichi, and Osaka.

This is a trimmed submission copy: `tests/`, `docs/`, and other dev-only
material have been removed, and only the code, configuration, dependency
definitions, and required source data needed to run the project are
included.

## Requirements

- Python >= 3.10
- Optional, only for the agentic-CLI conditions (C2/C3/C5–C12): the `codex`,
  `claude`, `agy` (Antigravity), or `opencode` CLIs, installed and
  authenticated.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Environment

```bash
cp .env.example .env
```

Fill in the keys you need:

- `GEMINI_API_KEY`, `OPENCODE_API_KEY` — required for the direct-API LLM
  conditions.
- `OPENROUTER_API_KEY` — optional additional provider.
- `ESTAT_APP_ID`, `REINFOLIB_API_KEY`, `YAHOO_CLIENT_ID` — only needed to
  refresh live source data (land prices, healthcare facility search); the
  pipeline degrades gracefully and notes what's missing if these are unset.

Load the environment before running any command:

```bash
set -a; source .env; set +a
```

Offline commands (`--help`, `doctor`, `validate-config`, `validate-views`,
`make pipeline`, `make validate`, `make audit`) work with no `.env` and no
network access at all.

`.env` must never be committed — it holds live secrets and is git-ignored.

## Data

`.data/manual/` ships with this submission (hospital cash-flow workbook and
population data) — it is the only required source input. Everything else
the pipeline needs is derived from it via `make pipeline`, or fetched live
via the optional API keys above.

## Run

```bash
make preflight              # probe configured live providers/harnesses
make pipeline                # build the deterministic data pipeline from .data/manual
make validate                # offline validation of configs, contracts, outputs
make audit                   # prototype-safety and eStat-usage audits
make reproduce-submission     # run conditions C0,C1,C4,C9,C10 (requires live agents)
```

### Running all C0–C14 conditions

All commands assume `set -a; source .env; set +a` has run, and share one
`--output-dir` (`OUT` below) so partial runs never overwrite conditions
already recorded.

```bash
OUT=outputs/condition_proposals/<run_name>
```

**Direct-API conditions** (C0, C1, C4, C13, C14 — one CLI
call each, no interactive session):

```bash
.venv/bin/python -m geo_strategist.cli run-condition-proposals \
  --conditions C0,C1,C4,C13,C14 \
  --output-dir "$OUT" --top-k-sites 5 --max-review-rounds 2 \
  --require-live-agents --disable-deterministic-fallback-for-comparison \
  --skip-judge
```

**Agentic-CLI conditions** (C2/C3/C5–C12 — need Codex / Claude Code /
Antigravity / OpenCode). Either drive them non-interactively:

```bash
.venv/bin/python -m geo_strategist.cli run-condition-proposals \
  --conditions C2,C3,C5,C6,C7,C8,C9,C10,C11,C12 \
  --output-dir "$OUT" --top-k-sites 5 --max-review-rounds 2 \
  --require-live-agents --disable-deterministic-fallback-for-comparison \
  --auto-agentic-harness --skip-judge
```

...or write handoff prompts and run each CLI session manually, then ingest
the result:

```bash
.venv/bin/python -m geo_strategist.cli write-manual-harness-prompts
# ... run the matching CLI (codex/claude/agy/opencode) with the prompt, save
# its JSON result to $OUT/runs/<Cxx>/manual_result.json, then:
.venv/bin/python -m geo_strategist.cli run-condition-proposals \
  --conditions C11 --output-dir "$OUT" \
  --manual-result "$OUT/runs/C11/manual_result.json" --skip-judge
```

C9–C12 read `AGENTS.md` and the `.agents/skills/` / `.claude/skills/`
packages during the session — both ship with this submission.

**Judge** (compares proposal quality and agentic-process quality across
conditions):

```bash
# Offline, deterministic checklist only:
.venv/bin/python -m geo_strategist.cli run-condition-comparison-judge \
  --proposals-dir "$OUT" --structural-judge-only

# Full run with the primary qualitative LLM-panel score:
.venv/bin/python -m geo_strategist.cli run-condition-comparison-judge \
  --proposals-dir "$OUT" --allow-live-judge
```

## Repository layout

- `src/geo_strategist/` — the package (CLI, providers, harnesses,
  experiments, agent/codeexec sandbox, data pipeline, reporting).
- `configs/` — canonical condition registry and runtime YAML config.
- `data/` — JSON-schema contracts and the source catalog.
- `schemas/` — runtime JSON schemas.
- `prompts/` — shared prompt templates.
- `scripts/` — pipeline, validation, and audit scripts used by the `Makefile`.
- `.agents/skills/`, `.claude/skills/` — Skills packages for the C9–C12
  conditions (`.claude/skills/` are symlinks into `.agents/skills/`).
- `.data/manual/` — required source input data (hospital workbook,
  population data).
- `AGENTS.md` — agent-workflow contract read by the agentic conditions.

## Status

Research prototype for comparing proposal-generation workflows. Every
generated report ends with a **Limitations / Required Due Diligence**
section listing what specialists must verify before any real-world action.
Proposals are decision-support drafts, never operationally final decisions.
