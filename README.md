# Geo Strategist

Geo Strategist is an evidence-graded experiment system for evaluating
**agentic hospital-location / hospital-reorganization / candidate-site-selection
proposal generation**. Fourteen conditions (C0–C13) generate proposals over the
same real local evidence base for Tokyo, Aichi, and Osaka (census population
projections, MLIT land prices, facility-search records, and a hospital
cash-flow workbook), and the condition-comparison judge compares proposal quality *and*
agentic-process quality across them.

Unlike earlier versions — which collapsed conditions into nearly identical
deterministic rankings — every LLM condition is a **real live-agent run**:
live API calls (Gemini direct, OpenCode Go) or coding-agent harness sessions
(Codex, Claude Code, OpenCode), with generated experiment code executed in a
sandbox, external data-grounded objective metrics steering all search, and
failure modes recorded honestly (never silently replaced by a deterministic
ranking).

## Canonical conditions

| id | label | provider / model | orchestration |
| --- | --- | --- | --- |
| C0 | Deterministic Python baseline | none | fixed weighted composite |
| C1 | Vanilla direct LLM baseline (Gemini) | gemini-3.5-flash | one call, no tools, no code (**strict pair with C9**) |
| C2 | Vanilla LLM baseline (Codex, no Skills) | Codex `codex` (subscription) / gpt-5.5 | one pass, no tools, no Skills (**strict pair with C6**) |
| C3 | Vanilla LLM baseline (Claude Code, no Skills) | Claude Code `claude` (subscription) / sonnet-5.0 | one pass, no tools, no Skills (**strict pair with C7**) |
| C4 | Vanilla direct LLM baseline (DeepSeek) | opencode_go / deepseek-v4-flash | one call, no tools, no code (**strict trio with C8/C10**) |
| C5 | Skills-Antigravity | Antigravity `agy` (subscription) / gemini-3.5-pro | Skills-unified + branch search (Skills-in-harness comparison C5–C8) |
| C6 | Skills-Codex | Codex `codex` (subscription) / gpt-5.5 | Skills-unified + branch search (**strict pair with C2**) |
| C7 | Skills-Claude Code | Claude Code `claude` (subscription) / sonnet-5.0 | Skills-unified + branch search (**strict pair with C3**) |
| C8 | Skills-OpenCode | OpenCode `opencode` (subscription) / deepseek-v4-flash | Skills-unified + branch search (**strict trio with C4/C10**) |
| C9 | Traditional AI Scientist-style Gemini | gemini-3.5-flash | AI-Scientist-v2-style loop (**strict pair with C1**; same family as C10) |
| C10 | Traditional AI Scientist-style DeepSeek | opencode_go / deepseek-v4-flash | AI-Scientist-v2-style loop, high throughput (**strict trio with C4/C8**) |
| C11 | Multi-model Shinka-style evolution | opencode_go multi-model | evolutionary strategy search |
| C12 | AB-MCTS-style adaptive branching | opencode_go multi-model | width/depth-adaptive program tree |
| C13 | Fugu-style dynamic orchestrator | opencode_go multi-model | router-assigned model roles |
| judge | Cross-condition comparison judge | gemini (+ optional OpenRouter) | structural + live LLM judge, 18 dimensions |

All branch/search conditions share five objectives: elderly-demand,
emergency-access, reorganization-feasibility, financial-risk,
evidence-completeness. See `docs/experiment_design.md` and
`docs/agentic_orchestration.md` (with cited sources: AI Scientist-v2,
ShinkaEvolve, AB-MCTS, Sakana Fugu).

## Evidence policy

- No fabricated hospital names, addresses, coordinates, parcels, facility
  facts, or financial values. Concrete fields carry provenance or an explicit
  grade (`not_available`, `model_estimate`, `scenario_assumption`,
  `unverified_candidate`).
- LLM slates are validated against the real candidate universe
  (`candidate_actions.jsonl`); fabricated candidate ids are rejected.
- Missing data never aborts a proposal; every gap lands in the closing
  **Limitations / Required Due Diligence** section of every report.
- Proposals are decision-support drafts, never operationally final decisions.

## Quick start

```bash
.venv/bin/python -m pip install -e ".[dev]"
set -a; source .env; set +a           # copy .env.example → .env first

# 1) Probe live providers and harnesses (writes outputs/provider_preflight/):
.venv/bin/python -m geo_strategist.cli check-live-agent-providers

# 2) Staged live experiment:
.venv/bin/python -m geo_strategist.cli run-condition-proposals \
  --conditions C0,C1,C4,C9,C10 \
  --output-dir outputs/condition_proposals/run_stage1 \
  --top-k-sites 5 --max-review-rounds 2 \
  --require-live-agents \
  --disable-deterministic-fallback-for-comparison \
  --branch-objectives elderly-demand,emergency-access,reorganization-feasibility,financial-risk,evidence-completeness
# stage2: --conditions C11,C12    stage3: --conditions C13
# harness conditions:
#   --conditions C2,C3        writes vanilla no-Skills handoff prompts
#   --conditions C5,C6,C7,C8  writes AGENTS.md + Skill package launchers

# Validation:
.venv/bin/python -m compileall src
.venv/bin/python -m pytest -q
```

Each run writes per-condition reports (`reports/CNN_<slug>.md` + figures + a
location map panel), candidate-level Qualitative Site Discussions with real
population/land-price figures,
`condition_records.jsonl`, `condition_outputs_summary.json`,
`run_manifest.json`, `artifact_index.md`, the
`reports/condition_comparison_report.md`, and per-condition trace directories
under `runs/<Cx>/` (journals, generated code, sandboxes, redacted model-call
traces). Full map: `docs/artifact_map.md`.

If Antigravity / Codex / Claude Code / OpenCode must be driven interactively,
the orchestrator writes C2/C3 vanilla handoff prompts and C5-C8 AGENTS.md +
Skill package launchers to `outputs/condition_proposals/live/manual_harness/`
and marks those conditions `waiting_for_manual_harness` — they are excluded
from the comparison until the manual results are ingested (see
`manual_harness/README.md` in the live output dir).

C5-C8 non-interactive CLI execution is opt-in with
`--auto-agentic-harness`. Unsupported adapters still fall back to launcher
mode and write redacted execution metadata/log placeholders under
`runs/<Cx>/`.

## Repository layout

- `configs/experiment_conditions.yaml` — canonical C0–C13 registry mirror.
- `docs/` — `experiment_design.md`, `agentic_orchestration.md`,
  `artifact_map.md`, `status.md`, plus data-policy context docs.
- `references/local/` — local reference material (git-ignored), including
  the AI Scientist-v2 source dump the C9/C10 loop adapts.
- `src/geo_strategist/providers/` — Gemini (Interactions API), OpenCode Go
  (chat completions with reasoning_content), OpenRouter clients + preflight.
- `src/geo_strategist/harnesses/` — Antigravity/Codex/Claude Code/OpenCode
  status checks, Skills launcher generation, and compatibility adapters.
- `src/geo_strategist/experiments/` — condition registry, branch objectives,
  live condition runners (`vanilla_llm`, `ai_scientist_loop`,
  `c11_shinka_evolution`, `c12_ab_mcts`,
  `c13_fugu_router`), deterministic engine (C0 + external metrics),
  orchestrator, condition-comparison judge, and the S1–S7/E14 site-selection track.
- `src/geo_strategist/agent/codeexec/` — sandbox guard + subprocess
  interpreter for generated code (adapted from AI Scientist-v2).
- `tests/` — data pipeline, contracts, providers, registry, offline
  fake-LLM condition runs, orchestrator, the comparison judge, and key-leak checks.

## Sandbox constraints for generated code

Generated code is statically scanned before execution (no subprocess, no
network imports, no environment reads, no file deletion, no absolute paths
outside the repo/run dir), then executed in a separate process with a
scrubbed environment (no API keys) and a hard timeout. Reasoning traces from
providers are stored as trace artifacts only and never pasted into final
reports.

## Environment

Copy `.env.example` to `.env` and fill in keys (`GEMINI_API_KEY`,
`OPENCODE_API_KEY`, optional `OPENROUTER_API_KEY`; `ESTAT_APP_ID` /
`REINFOLIB_API_KEY` / `YAHOO_CLIENT_ID` for live data refresh). `.env` stays
local-only; never commit secrets, raw API outputs, caches, or run outputs.

## Status

Research prototype for comparing proposal-generation workflows. Every report
ends with a **Limitations / Required Due Diligence** section listing what
specialists must verify before any real-world action. See `docs/status.md`.
