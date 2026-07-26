"""Manual handoff prompt files for interactively-driven harness conditions.

Three families require an interactive subscription session and a written
handoff prompt:

- C2 (Codex, no Skills) and C3 (Claude Code, no Skills) — vanilla,
  single-pass baselines that solve the task with the harness's native
  abilities, no Skills contract, no branch search.
- C5 (Antigravity), C6 (Codex), C7 (Claude Code), C8 (OpenCode) — coding-agent
  controls: a full native coding-agent session (repository inspection,
  editing, code execution, debugging) but with project Skills deliberately
  withheld, so the delta against their C9-C12 strict pair isolates the
  Skills contract rather than "having a coding agent at all".
- C9 (Antigravity), C10 (Codex), C11 (Claude Code), C12 (OpenCode) — the
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
from geo_strategist.experiments.decision_reporting_contract import (
    reporting_prompt_fragment,
    reporting_schema_example,
)

DEFAULT_LIVE_DIR = Path("outputs/condition_proposals/live")

_MANUAL_CONDITIONS = ("C2", "C3", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12")


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
    "Use the Skills-unified packages as a dynamic operator library. Select and "
    "reuse Skills based on evidence gaps, execution outcomes, critique, and unresolved "
    "decision risk. Cover the five anchors below, but generate additional materially "
    "different regimes when supported. Execute alternatives over the full candidate "
    "universe unless a recorded eligibility rule narrows it, then validate, review, "
    "revise, stress test, and synthesize decision portfolios."
)

_VANILLA_NO_SKILLS_DEFINITION = (
    "Pure single-pass vanilla condition — do NOT follow the Skills-unified "
    "contract, do NOT run a branch search, and do NOT write or execute any "
    "analysis code. Read the data directly and produce a single ranked "
    "slate from your own reasoning in one pass, with no other tool use. "
    "This is the no-Skills vanilla baseline on this harness/model — leave "
    "`skill_trace` and any generated_code directory empty."
)

_CODING_AGENT_NO_SKILLS_DEFINITION = (
    "Coding-agent control condition — a full native coding-agent session: "
    "you may freely inspect the repository, read data files, write and "
    "iterate on analysis code, execute it, and debug failures across as "
    "many steps as you judge necessary. The ONLY thing withheld is the "
    "project-specific Skills contract: do NOT read or follow "
    "`.agents/skills/`, `.claude/skills/`, or any Skills-unified procedure, "
    "or invoke them. Conduct open-ended hypothesis, regime, code, validation, "
    "robustness, review, revision, and portfolio search with your own native "
    "agentic judgment. The five objectives below are suggested analytical "
    "lenses, not a required checklist or a fixed branch pipeline — covering "
    "one, several, or none of them while pursuing your own regimes is a "
    "valid outcome. This isolates the "
    "Skills contract as the sole treatment difference against this "
    "condition's Skills-unified strict pair (same provider, model, and "
    "harness). Its `skill_trace` in the return format is optional; leave it "
    "empty or list the major steps you actually took."
)

_CONDITION_DEFS: dict[str, dict[str, object]] = {
    "C2": {
        "title": "C2 — Vanilla LLM baseline (Codex, no Skills)",
        "harness_command": "codex  # interactive session in the repository root",
        "definition": _VANILLA_NO_SKILLS_DEFINITION + (
            " Part of the C1-C4 vanilla-baseline comparison; strict pair with "
            "C10 (same Codex/gpt-5.5 stack, Skills-unified + branch search)."
        ),
        "branch_search": False,
    },
    "C3": {
        "title": "C3 — Vanilla LLM baseline (Claude Code, no Skills)",
        "harness_command": "claude  # interactive session in the repository root",
        "definition": _VANILLA_NO_SKILLS_DEFINITION + (
            " Part of the C1-C4 vanilla-baseline comparison; strict pair with "
            "C11 (same Claude Code/sonnet-5.0 stack, Skills-unified + branch search)."
        ),
        "branch_search": False,
    },
    "C5": {
        "title": "C5 — Antigravity coding-agent control",
        "harness_command": "agy  # interactive Antigravity session in the repository root",
        "definition": _CODING_AGENT_NO_SKILLS_DEFINITION + (
            " Strict pair with C9 (same Antigravity/gemini-3.5-flash stack, "
            "Skills-unified + branch search)."
        ),
        "branch_search": True,
    },
    "C6": {
        "title": "C6 — Codex coding-agent control",
        "harness_command": "codex  # interactive session in the repository root",
        "definition": _CODING_AGENT_NO_SKILLS_DEFINITION + (
            " Strict pair with C10 (same Codex/gpt-5.5 stack, Skills-unified "
            "+ branch search)."
        ),
        "branch_search": True,
    },
    "C7": {
        "title": "C7 — Claude Code coding-agent control",
        "harness_command": "claude  # interactive session in the repository root",
        "definition": _CODING_AGENT_NO_SKILLS_DEFINITION + (
            " Strict pair with C11 (same Claude Code/sonnet-5.0 stack, "
            "Skills-unified + branch search)."
        ),
        "branch_search": True,
    },
    "C8": {
        "title": "C8 — OpenCode coding-agent control",
        "harness_command": (
            "opencode --model opencode_go/deepseek-v4-flash  # interactive session in the repository root"
        ),
        "definition": _CODING_AGENT_NO_SKILLS_DEFINITION + (
            " Strict pair with C12 (same OpenCode/deepseek-v4-flash stack, "
            "Skills-unified + branch search)."
        ),
        "branch_search": True,
    },
    "C9": {
        "title": "C9 — Skills-Antigravity",
        "harness_command": "agy  # interactive Antigravity session in the repository root",
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "Antigravity harness (Gemini). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C9-C12 Skills-in-harness comparison; only the "
            "harness/model stack differs between those four conditions."
        ),
        "branch_search": True,
    },
    "C10": {
        "title": "C10 — Skills-Codex",
        "harness_command": "codex  # interactive session in the repository root",
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "Codex harness (must stay on gpt-5.5 for the strict C2/C10 "
            "comparison). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C9-C12 Skills-in-harness comparison."
        ),
        "branch_search": True,
    },
    "C11": {
        "title": "C11 — Skills-Claude Code",
        "harness_command": "claude  # interactive session in the repository root",
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "Claude Code harness (must stay on sonnet-5.0 for the strict "
            "C3/C11 comparison). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C9-C12 Skills-in-harness comparison."
        ),
        "branch_search": True,
    },
    "C12": {
        "title": "C12 — Skills-OpenCode",
        "harness_command": (
            "opencode --model opencode_go/deepseek-v4-flash  # interactive session in the repository root"
        ),
        "definition": (
            "Skills-unified coding-agent condition with branch search on the "
            "OpenCode harness with provider opencode_go and model "
            "deepseek-v4-flash (must match C4/C14 exactly for the strict "
            "C4/C12/C14 comparison). " + _SKILLS_CONTRACT_WITH_BRANCH
            + " Part of the C9-C12 Skills-in-harness comparison."
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

_C0_SUBSTITUTION_RULES = """\
## Do not substitute the deterministic baseline (mandatory)

Your final ranking must come from your own analysis, not from the project's
deterministic reference. Specifically, do NOT use any of the following to
select or order your final slate:

- the C0 report or run directory (`outputs/condition_proposals/live/runs/C00/`);
- the C0 ranked-candidate artifact (`ranked_candidates.jsonl`) or C0's final
  slate;
- `deterministic_evaluation_engine.py` (or any import from it) as your final
  ranking procedure;
- the default composite score, or `priority_score` alone, as your final
  ranking criterion;
- a copied or trivially reordered version of the deterministic baseline
  ranking.

You may inspect shared raw evidence fields and reusable data-loading
utilities (the same underlying datasets C0 also reads) — that is not a
violation. The violation is using C0's *ranking output or ranking method* in
place of your own. If your final slate happens to overlap with C0's, that is
only acceptable when you can point to your own executed analysis (generated
code, recorded reasoning) that independently arrived at it — never present
an unexplained match as if it were your own method. Explain in your final
rationale/review_comments how your evaluation design differs from the
default deterministic composite.
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


def _objectives_block(*, required: bool = True) -> str:
    if required:
        lines = ["## Minimum strategy-coverage anchors (additional regimes allowed)", ""]
    else:
        # C5-C8 (open-ended native-agent controls): these are lenses to
        # consider, not a checklist the run is validated against — the
        # native_agent_model family contract requires only a minimum
        # execution surface (valid slate, traceable artifacts, a final
        # rationale), never coverage of all five. Presenting them as
        # "minimum" anchors here would contradict that contract and imply a
        # requirement the harness is not actually held to.
        lines = [
            "## Suggested analytical lenses (not required; explore freely)",
            "",
            "You are not required to cover all five, and covering only one or "
            "adding others of your own is a valid outcome:",
            "",
        ]
    for objective in BRANCH_OBJECTIVES:
        lines.append(f"- `{objective.key}` ({objective.label}): {objective.description}")
    return "\n".join(lines) + "\n"


def _save_path_for_prompt(repo_root: Path, live_dir: Path) -> str:
    """Path used in the agent-facing save instruction.

    A path relative to the repository root, not an absolute path: a
    sandboxed coding-agent session (e.g. C5-C8's isolated no-Skills
    workspace, or any CLI's own workspace-write sandbox) may enforce its
    writable boundary against its own working directory rather than the
    real filesystem root, so an absolute real-repo path can be rejected
    even though the agent's cwd mirrors the same directory structure
    (via symlinks) and a relative path resolves correctly either way.
    """

    try:
        return str(live_dir.relative_to(repo_root))
    except ValueError:
        return str(live_dir)


def _workspace_block(
    repo_root: Path,
    harness_command: str,
    *,
    sanitized_workspace: Path | None,
) -> str:
    """Where to work: the real repo root for a human-run interactive session,
    or the sanitized no-Skills workspace for an automated C5-C8 run.

    The two must never be conflated: telling an automated run (whose cwd is
    already the sanitized workspace) to "run from the repository root" using
    the real, unsanitized absolute path contradicts the isolation the run
    depends on — this was found live in an actual C5 run's generated prompt.
    """

    if sanitized_workspace is not None:
        bare_command = harness_command.split("#", 1)[0].strip()
        return (
            f"Real repository root (for reference only — do NOT navigate "
            f"here via an absolute path or `cd`): `{repo_root}`\n\n"
            "This is an automated, physically isolated run. Your current "
            "working directory is already set to a sanitized workspace that "
            "mirrors the real repository via symlinks, with the project "
            f"Skills packages physically removed: `{sanitized_workspace}`\n\n"
            "Rules for this session:\n\n"
            "- Treat your current working directory as the repository root. "
            "Use ONLY paths relative to it for every read and write (e.g. "
            "`.data/interim/...`, `outputs/condition_proposals/live/runs/...`).\n"
            f"- Do NOT `cd` to `{repo_root}` or any other absolute path "
            "outside your current working directory — that would escape "
            "the sanitized workspace and defeat the isolation this run "
            "depends on.\n"
            "- Do NOT read `.agents/skills/` or `.claude/skills/` — they are "
            "not reachable from this workspace, and must stay that way.\n"
            "- Do NOT follow any Skills-unified procedure.\n\n"
            "You are already running inside the sanitized workspace; no "
            f"further invocation is needed (informational command: "
            f"`{bare_command}`).\n"
        )
    return (
        f"Repository: `{repo_root}`\n"
        "Run from the repository root. Suggested invocation:\n\n"
        f"```\n{harness_command}\n```\n"
    )


def _completion_loop_block(save_root: str, padded: str) -> str:
    validator_command = (
        f".venv/bin/python -m geo_strategist.cli validate-c5-result "
        f"{save_root}/runs/{padded}/manual_result.json"
    )
    return f"""## Validate and repair before you stop (mandatory)

After you write `manual_result.json`, run the validator and repair any
reported issue before finishing — do not stop at "wrote the file":

```
{validator_command}
```

1. Write `manual_result.json`.
2. Run the exact validator command above.
3. Read every reported error and every C0-substitution flag (a
   C0-substitution flag is never acceptable and must be fixed, not
   explained away).
4. Repair the output, or the supporting `generated_code/` that produced it.
5. Re-run the validator.
6. Repeat steps 3-5 for up to 3 repair attempts.
7. Stop only once the validator reports PASS, or — if it still fails after
   3 repair attempts — write an explicit failure note explaining what still
   fails and why, to
   `{save_root}/runs/{padded}/unresolved_validation_failure.md`, rather than
   leaving a silently invalid `manual_result.json` as your only output.

Also run the compile/test commands above at least once before finishing.
"""


def _return_format_block(condition: str, live_dir: Path, *, save_root: str) -> str:
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
    schema.update(reporting_schema_example(condition))
    return (
        "## Required return format\n\n"
        f"Save exactly one JSON file to `{save_root}/runs/{padded}/manual_result.json`\n"
        "(a path relative to the repository root you were given — write it there "
        "even if your working directory is a sandboxed mirror of the repository; "
        "do NOT use an absolute path from a different session, which a sandbox may reject):\n\n"
        "```json\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n```\n\n"
        "Rules: `candidate_id` values must come verbatim from "
        "`.data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl` "
        "(unknown ids are rejected at ingestion); provide at least 5 ranked "
        "candidates; every candidate needs the full `qualitative_discussion` "
        "object. For C5-C8, keep any generated code or execution artifacts under "
        f"`{save_root}/runs/{padded}/generated_code/`; C2-C3 must not create or execute code.\n"
    )


def build_manual_prompt(
    condition: str,
    repo_root: Path,
    live_dir: Path,
    *,
    sanitized_workspace: Path | None = None,
) -> str:
    from geo_strategist.experiments.condition_registry import build_condition_registry
    from geo_strategist.harnesses.agentic_runner import build_agentic_launcher_prompt

    spec_obj = build_condition_registry()[condition]
    if spec_obj.runner == "agentic_skills_harness":
        return build_agentic_launcher_prompt(spec_obj, repo_root, live_dir)

    spec = _CONDITION_DEFS[condition]
    padded = _padded(condition)
    save_root = _save_path_for_prompt(repo_root, live_dir)
    is_no_skills_control = spec_obj.runner == "coding_agent_no_skills"
    workspace_block = _workspace_block(
        repo_root, spec["harness_command"],
        sanitized_workspace=sanitized_workspace if is_no_skills_control else None)
    substitution_block = f"\n{_C0_SUBSTITUTION_RULES}" if is_no_skills_control else ""
    completion_block = (
        f"\n{_completion_loop_block(save_root, padded)}" if is_no_skills_control else ""
    )
    return f"""# Manual harness handoff — {spec['title']}

{workspace_block}
## Condition definition

{spec['definition']}

{_objectives_block(required=spec_obj.primary_comparison_family != "native_agent_model")}
{_QUALITATIVE_REQUIREMENT}
{reporting_prompt_fragment(condition)}
{_return_format_block(condition, live_dir, save_root=save_root)}
{_EVIDENCE_RULES}
{substitution_block}
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
{completion_block}
## When you are done

Stop here. Do NOT run `run-condition-proposals` or
`run-condition-comparison-judge` yourself — an orchestrator ingests
`manual_result.json` and refreshes the comparison automatically once every
condition in the batch has finished. Running these yourself from inside
this session can race with, or block, that orchestration.
"""


def write_manual_harness_prompts(
    repo_root: str | Path = ".",
    conditions: list[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    sanitized_workspace: str | Path | None = None,
) -> dict[str, Path]:
    """Write handoff prompts. ``sanitized_workspace`` (the C5-C8 automated
    run's isolated working directory) is only applied when exactly one
    condition is requested — it is meaningless across a batch write."""

    root = Path(repo_root).resolve()
    live_dir = Path(output_dir) if output_dir else root / DEFAULT_LIVE_DIR
    if not live_dir.is_absolute():
        live_dir = root / live_dir
    prompts_dir = live_dir / "manual_harness"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    prompt_files = _prompt_files()
    target_conditions = conditions or list(prompt_files)
    workspace_path = Path(sanitized_workspace) if sanitized_workspace is not None else None
    apply_workspace = workspace_path is not None and len(target_conditions) == 1
    for condition in target_conditions:
        if condition not in prompt_files:
            continue
        path = prompts_dir / prompt_files[condition]
        path.write_text(
            build_manual_prompt(
                condition, root, live_dir,
                sanitized_workspace=workspace_path if apply_workspace else None),
            encoding="utf-8")
        written[condition] = path
    return written


def write_manual_harness_readme(live_dir: str | Path) -> Path:
    """The manual-harness README, generated next to the prompts so its
    commands and filenames come from the same source as the prompt files."""

    live_dir = Path(live_dir)
    files = _prompt_files()
    text = f"""# Manual and Skills harness execution (C2 / C3 / C5-C8 / C9-C12)

Three families run inside interactive coding-agent sessions:

- Vanilla, no-Skills baselines: `{files['C2']}` (C2, Codex) and
  `{files['C3']}` (C3, Claude Code) — single pass, no tool use, no Skills
  contract, no branch search.
- Coding-agent, no-Skills controls: `{files['C5']}` (C5, Antigravity),
  `{files['C6']}` (C6, Codex), `{files['C7']}` (C7, Claude Code), and
  `{files['C8']}` (C8, OpenCode) — full native coding-agent sessions
  (multi-step repository inspection, editing, code execution, debugging)
  with project Skills deliberately withheld.
- AGENTS.md + Skills harness conditions: `{files['C9']}` (C9, Antigravity via `agy`),
  `{files['C10']}` (C10, Codex), `{files['C11']}` (C11, Claude Code), and
  `{files['C12']}` (C12, OpenCode) — short launcher prompts that direct the
  agent to read `AGENTS.md` and the filesystem Skill packages before running
  the full Skills-unified contract plus branch search.

Until you run them, they stay `waiting_for_manual_harness` and are excluded
from the comparison. C5/C9, C6/C10, C7/C11, and C8/C12 are the four strict
pairs isolating the Skills contract (same provider/model/harness on each
side). C2/C10 also share the Codex/gpt-5.5 stack; C3/C11 also share the
Claude Code/sonnet-5.0 stack; C12 must stay on opencode_go/deepseek-v4-flash
so the strict C4/C8/C12/C14 comparisons hold.

```bash
# 1. Open the prompt (always freshly regenerated from source — run
#    write-manual-harness-prompts first if you are not sure it is current)
cat {live_dir}/manual_harness/{files['C2']}

# 2. Run it manually in the matching CLI (codex / claude / agy / opencode),
#    from the repo root. C9-C12 launchers require AGENTS.md + Skill packages.
#    For the Antigravity conditions (C5, C9), a bounded smoke test is
#    available before committing to a long session:
#    .venv/bin/python -m geo_strategist.cli antigravity-preflight

# 3. Save the returned result exactly where the prompt says, e.g.:
#    {live_dir}/runs/{_padded('C2')}/manual_result.json

# 4. Validate before ingesting — repair any reported issue and re-run this
#    until it passes. C5-C8 (no-Skills controls) use validate-c5-result;
#    C9-C12 (Skills-unified) use validate-skills-result with the matching
#    --expected-condition-group:
.venv/bin/python -m geo_strategist.cli validate-c5-result \\
  {live_dir}/runs/{_padded('C5')}/manual_result.json
.venv/bin/python -m geo_strategist.cli validate-skills-result \\
  {live_dir}/runs/{_padded('C9')}/manual_result.json --expected-condition-group C9

# 5. Ingest the manual result (updates only that condition's record)
.venv/bin/python -m geo_strategist.cli run-condition-proposals \\
  --conditions C2 --output-dir {live_dir} \\
  --manual-result {live_dir}/runs/{_padded('C2')}/manual_result.json --skip-judge

# 6. Re-run the condition comparison over all ingested records
.venv/bin/python -m geo_strategist.cli run-condition-comparison-judge \\
  --proposals-dir {live_dir}
```

Repeat steps 1-5 with `{files['C3']}` (C3, Claude Code), `{files['C5']}` (C5,
Antigravity control), `{files['C6']}` (C6, Codex control), `{files['C7']}` (C7,
Claude Code control), `{files['C8']}` (C8, OpenCode control), `{files['C9']}`
(C9, Antigravity), `{files['C10']}` (C10, Codex), `{files['C11']}` (C11, Claude
Code), and `{files['C12']}` (C12, OpenCode). The `manual_result.json` schema is
embedded in each prompt or launcher; candidate_ids must come from
`.data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl`,
every candidate needs the seven-part `qualitative_discussion`, and
deterministic fallback rankings are never acceptable as harness output — do
not use a C0 report, C0's ranked output, `deterministic_evaluation_engine.py`,
or the default composite score as your final ranking procedure (both
validators flag likely substitution; see each condition's prompt for the
full list). C9-C12 runs also need a complete `skill_trace`. A skill-trace
row that claims executed analysis with no resolvable output behind it (a
broken or missing output reference) is an unsupported factual claim and
excludes the run from the condition-comparison judge; other trace-shape or
lifecycle differences are recorded as deviations and do not exclude the
run. C5-C8 must NOT install, expose, or use project Skills — do not read
`.agents/skills/` or `.claude/skills/` during these sessions.

An `--auto-agentic-harness` unattended option also exists for
`run-condition-proposals` (adds `--auto-agentic-harness` to the command in
step 5); for C5/C9 specifically it runs a real, physically isolated
automated session instead of this manual walkthrough. Prefer the manual
walkthrough above when you want to review/approve steps yourself, or if
another interactive `agy` session is already open on the same condition —
the automated path and a concurrent interactive session can share
underlying workspace/project state and should not be run at the same time.
"""
    path = live_dir / "manual_harness" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
