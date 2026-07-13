"""Manual handoff prompt files for interactively-driven harness conditions.

Two families require an interactive subscription session and a written
handoff prompt:

- C2 (Codex, no Skills) and C3 (Claude Code, no Skills) — vanilla,
  single-pass baselines that solve the task with the harness's native
  abilities, no Skills contract, no branch search.
- C5 (Antigravity), C6 (Codex), C7 (Claude Code), C8 (OpenCode) — the
  AGENTS.md + filesystem Skills harness conditions. These receive short
  launcher prompts; the Skill bodies live in ``.agents/skills`` and
  ``.claude/skills``.

For each such condition a complete, self-contained prompt file is written
under ``<output_dir>/manual_harness/`` (default
``outputs/condition_proposals/live/manual_harness/``); the condition is
marked ``waiting_for_manual_harness`` until the harness's
``manual_result.json`` is ingested via ``--manual-result``.
"""

from __future__ import annotations

import json
from pathlib import Path

from geo_strategist.experiments.branch_objectives import BRANCH_OBJECTIVES

DEFAULT_LIVE_DIR = Path("outputs/condition_proposals/live")

_MANUAL_CONDITIONS = ("C2", "C3", "C5", "C6", "C7", "C8")


def _padded(condition: str) -> str:
    return f"C{int(condition[1:]):02d}"


def prompt_filename(condition: str) -> str:
    """Prompt basename derived from the registry report slug, so the prompt,
    report, and run-directory names cannot drift apart."""

    from geo_strategist.experiments.condition_registry import build_condition_registry

    spec = build_condition_registry()[condition]
    slug = spec.report_slug
    if spec.runner == "agentic_skills_harness":
        from geo_strategist.harnesses.agentic_runner import launcher_filename

        return launcher_filename(spec)
    return f"{slug.removesuffix('_manual')}_prompt.md"


def _prompt_files() -> dict[str, str]:
    return {condition: prompt_filename(condition) for condition in _MANUAL_CONDITIONS}


_SKILLS_CONTRACT_WITH_BRANCH = (
    "Walk the Skills-unified contract: "
    "inspect_available_data → generate_research_hypotheses → design_evaluation_model → "
    "write_experiment_code → execute_generated_code → debug_failed_code → "
    "run_branch_search (explore one branch per objective listed below, generating "
    "and executing a scoring-code variant per branch) → review_proposal → "
    "revise_proposal → write_final_condition_proposal. Synthesize the final "
    "slate from the branch winners."
)

_VANILLA_NO_SKILLS_DEFINITION = (
    "Pure interactive-harness condition — do NOT follow the Skills-unified "
    "contract and do NOT run a branch search. Solve the task with the "
    "harness's native agentic abilities (inspect the data, write and execute "
    "your own analysis code if you choose to, and produce a single ranked "
    "slate) in whatever order you judge best, in one pass. This is the "
    "no-Skills vanilla baseline on this harness/model — its `skill_trace` in "
    "the return format is optional; leave it empty or list the major steps "
    "you actually took."
)

_CONDITION_DEFS: dict[str, dict[str, object]] = {
    "C2": {
        "title": "C2 — Vanilla LLM baseline (Codex, no Skills)",
        "harness_command": "codex  # interactive session in the repository root",
        "definition": _VANILLA_NO_SKILLS_DEFINITION + (
            " Part of the C1-C4 vanilla-baseline comparison; strict pair with "
            "C6 (same Codex/gpt-5.5 stack, Skills-unified + branch search)."
        ),
        "branch_search": False,
    },
    "C3": {
        "title": "C3 — Vanilla LLM baseline (Claude Code, no Skills)",
        "harness_command": "claude  # interactive session in the repository root",
        "definition": _VANILLA_NO_SKILLS_DEFINITION + (
            " Part of the C1-C4 vanilla-baseline comparison; strict pair with "
            "C7 (same Claude Code/sonnet-5.0 stack, Skills-unified + branch search)."
        ),
        "branch_search": False,
    },
    "C5": {
        "title": "C5 — Skills-Antigravity",
        "harness_command": "agy  # interactive Antigravity session in the repository root",
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "Antigravity harness (Gemini). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C5-C8 Skills-in-harness comparison; only the "
            "harness/model stack differs between those four conditions."
        ),
        "branch_search": True,
    },
    "C6": {
        "title": "C6 — Skills-Codex",
        "harness_command": "codex  # interactive session in the repository root",
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "Codex harness (must stay on gpt-5.5 for the strict C2/C6 "
            "comparison). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C5-C8 Skills-in-harness comparison."
        ),
        "branch_search": True,
    },
    "C7": {
        "title": "C7 — Skills-Claude Code",
        "harness_command": "claude  # interactive session in the repository root",
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "Claude Code harness (must stay on sonnet-5.0 for the strict "
            "C3/C7 comparison). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C5-C8 Skills-in-harness comparison."
        ),
        "branch_search": True,
    },
    "C8": {
        "title": "C8 — Skills-OpenCode",
        "harness_command": (
            "opencode --model opencode_go/deepseek-v4-flash  # interactive session in the repository root"
        ),
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "OpenCode harness with provider opencode_go and model "
            "deepseek-v4-flash (must match C4/C10 exactly for the strict "
            "C4/C8/C10 comparison). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C5-C8 Skills-in-harness comparison."
        ),
        "branch_search": True,
    },
}

_EVIDENCE_RULES = """\
## Evidence and provenance rules (mandatory)

- Do NOT fabricate hospital names, addresses, coordinates, land parcels,
  facility facts, financial values, travel times, or regulatory constraints.
- Every concrete field must carry provenance (a source reference into the
  local datasets) or one of: `not_available`, `model_estimate`,
  `scenario_assumption`, `unverified_candidate`.
- Missing data must not stop proposal generation; state it as unconfirmed and
  list it as a due-diligence item.
- Do NOT substitute deterministic fallback rankings for live agent output. If
  you cannot complete a step, record the failure explicitly; never silently
  copy the C0 baseline ranking.
- Output is decision support requiring human specialist verification, never an
  operationally final hospital-location decision.
"""

_QUALITATIVE_REQUIREMENT = """\
## Qualitative site discussion (required per candidate)

For EVERY candidate in your final slate, write a short qualitative discussion
covering all seven dimensions below, in prose, using only what the local data
supports (state anything else as unconfirmed):

1. `regional` — urban / suburban / aging / medical-cluster / underserved /
   reorganization-potential characterization.
2. `population` — total population, 65+ population and share, projected trend
   or decline risk (census projection files); say "unconfirmed" when missing.
3. `demand_supply` — demand score, supply-shortage score, relationship with
   existing facilities; name nearby hospitals ONLY from source-traceable
   records.
4. `access` — elderly demand and emergency-access considerations; distance is
   only a proxy, never assert travel times.
5. `cost_financial` — land component, financial component, cash-flow-workbook
   plausibility; never invent land/acquisition/construction/renovation costs;
   state explicitly that cost assessment and zoning checks are required.
6. `preferred_action` — build / reorganize / consolidate / expand / defer,
   grounded in the validated candidate action.
7. `review_comments` — main issues your own review raised (insufficient
   evidence, unverified travel time, unverified regulation, unverified
   financial feasibility, incomplete nearby-facility data).
"""


def _objectives_block() -> str:
    lines = ["## Branch objectives (use exactly these five)", ""]
    for objective in BRANCH_OBJECTIVES:
        lines.append(f"- `{objective.key}` ({objective.label}): {objective.description}")
    return "\n".join(lines) + "\n"


def _return_format_block(condition: str, live_dir: Path) -> str:
    padded = _padded(condition)
    schema = {
        "condition_group": condition,
        "ranked_candidates": [{
            "candidate_id": "<exact id from candidate_actions.jsonl>",
            "rationale": "<why this candidate, <= 60 words>",
            "qualitative_discussion": {
                "regional": "...", "population": "...", "demand_supply": "...",
                "access": "...", "cost_financial": "...",
                "preferred_action": "...", "review_comments": "...",
            },
        }],
        "review_comments": ["<slate-level review findings>"],
        "skill_trace": [{"skill_id": "inspect_available_data", "status": "succeeded"}],
        "model_call_summary": {"total_requests": 0},
    }
    return (
        "## Required return format\n\n"
        f"Save exactly one JSON file to `{live_dir}/runs/{padded}/manual_result.json`:\n\n"
        "```json\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n```\n\n"
        "Rules: `candidate_id` values must come verbatim from "
        "`.data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl` "
        "(unknown ids are rejected at ingestion); provide at least 5 ranked "
        "candidates; every candidate needs the full `qualitative_discussion` "
        "object; also keep the generated code you executed under "
        f"`{live_dir}/runs/{padded}/generated_code/`.\n"
    )


def build_manual_prompt(condition: str, repo_root: Path, live_dir: Path) -> str:
    from geo_strategist.experiments.condition_registry import build_condition_registry
    from geo_strategist.harnesses.agentic_runner import build_agentic_launcher_prompt

    spec_obj = build_condition_registry()[condition]
    if spec_obj.runner == "agentic_skills_harness":
        return build_agentic_launcher_prompt(spec_obj, repo_root, live_dir)

    spec = _CONDITION_DEFS[condition]
    padded = _padded(condition)
    return f"""# Manual harness handoff — {spec['title']}

Repository: `{repo_root}`
Run from the repository root. Suggested invocation:

```
{spec['harness_command']}
```

## Condition definition

{spec['definition']}

{_objectives_block()}
{_QUALITATIVE_REQUIREMENT}
{_return_format_block(condition, live_dir)}
{_EVIDENCE_RULES}

## Data to use

All ranking inputs live under `.data/interim/study_area/tokyo_aichi_osaka/`:
`candidate_actions.jsonl` (the only allowed candidate universe),
`municipality_scores_enriched.jsonl`, `municipality_feature_base.jsonl`,
`population_features.jsonl` (population totals and 65+ by year),
`municipality_land_features.jsonl` (MLIT land-price medians),
`municipality_healthcare_supply_features.jsonl` (facility counts),
`hospital_features.jsonl` (cash-flow workbook model estimates), and
`candidate_evidence_bundles.jsonl`.
Facility targets may only be named from source-traceable facility records.

## Compile / test commands

```
.venv/bin/python -m compileall src
.venv/bin/python -m pytest -q
```

## When you are done

Ingest the result and refresh the comparison:

```
.venv/bin/python -m geo_strategist.cli run-condition-proposals \\
  --conditions {condition} --output-dir {live_dir} \\
  --manual-result {live_dir}/runs/{padded}/manual_result.json --skip-judge
.venv/bin/python -m geo_strategist.cli run-condition-comparison-judge \\
  --proposals-dir {live_dir}
```
"""


def write_manual_harness_prompts(
    repo_root: str | Path = ".",
    conditions: list[str] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    live_dir = Path(output_dir) if output_dir else root / DEFAULT_LIVE_DIR
    if not live_dir.is_absolute():
        live_dir = root / live_dir
    prompts_dir = live_dir / "manual_harness"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    prompt_files = _prompt_files()
    for condition in conditions or list(prompt_files):
        if condition not in prompt_files:
            continue
        path = prompts_dir / prompt_files[condition]
        path.write_text(build_manual_prompt(condition, root, live_dir), encoding="utf-8")
        written[condition] = path
    return written


def write_manual_harness_readme(live_dir: str | Path) -> Path:
    """The manual-harness README, generated next to the prompts so its
    commands and filenames come from the same source as the prompt files."""

    live_dir = Path(live_dir)
    files = _prompt_files()
    text = f"""# Manual and Skills harness execution (C2 / C3 / C5 / C6 / C7 / C8)

Two families run inside interactive coding-agent sessions:

- Vanilla, no-Skills baselines: `{files['C2']}` (C2, Codex) and
  `{files['C3']}` (C3, Claude Code) — single pass, no Skills contract, no
  branch search.
- AGENTS.md + Skills harness conditions: `{files['C5']}` (C5, Antigravity via `agy`),
  `{files['C6']}` (C6, Codex), `{files['C7']}` (C7, Claude Code), and
  `{files['C8']}` (C8, OpenCode) — short launcher prompts that direct the
  agent to read `AGENTS.md` and the filesystem Skill packages before running
  the full Skills-unified contract plus branch search.

Until you run them, they stay `waiting_for_manual_harness` and are excluded
from the comparison. C2/C6 share the Codex/gpt-5.5 stack (strict pair);
C3/C7 share the Claude Code/sonnet-5.0 stack (strict pair); C8 must stay on
opencode_go/deepseek-v4-flash so the strict C4/C8/C10 comparison holds.

```bash
# 1. Open the prompt
cat {live_dir}/manual_harness/{files['C2']}

# 2. Run it manually in the matching CLI (codex / claude / agy / opencode),
#    from the repo root. C5-C8 launchers require AGENTS.md + Skill packages.

# 3. Save the returned result exactly where the prompt says, e.g.:
#    {live_dir}/runs/{_padded('C2')}/manual_result.json

# 4. Ingest the manual result (updates only that condition's record)
.venv/bin/python -m geo_strategist.cli run-condition-proposals \\
  --conditions C2 --output-dir {live_dir} \\
  --manual-result {live_dir}/runs/{_padded('C2')}/manual_result.json --skip-judge

# 5. Re-run the condition comparison over all ingested records
.venv/bin/python -m geo_strategist.cli run-condition-comparison-judge \\
  --proposals-dir {live_dir}
```

Repeat steps 1-4 with `{files['C3']}` (C3, Claude Code), `{files['C5']}` (C5,
Antigravity), `{files['C6']}` (C6, Codex), `{files['C7']}` (C7, Claude Code),
and `{files['C8']}` (C8, OpenCode). The `manual_result.json` schema is
embedded in each prompt or launcher; candidate_ids must come from
`.data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl`,
every candidate needs the seven-part `qualitative_discussion`, and
deterministic fallback rankings are never acceptable as harness output.
C5-C8 runs also need a complete valid `skill_trace`; invalid Skills traces are
excluded from the condition-comparison judge.
"""
    path = live_dir / "manual_harness" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
