"""Antigravity adapter metadata for Skills launcher mode.

``--print`` (aka ``--prompt``/``-p``) consumes only the single token that
immediately follows it as the prompt text — confirmed live against the
installed CLI (agy 1.1.5): placing any other flag between ``--print`` and
the prompt (e.g. ``agy --print --print-timeout 110m <prompt>``) makes the
CLI swallow the literal string ``"--print-timeout"`` as the prompt instead,
and the model then free-associates about that flag itself (reads its own
CLI reference docs, tries to run `agy --print-timeout`, etc.) rather than
doing the actual task — this exact failure mode was previously seen and
noted as "silently ignores the prompt text" without the mechanism being
understood. The fix is ordering: the command template here is only
``agy --print`` so the prompt lands immediately after it (appended by
``harnesses/agentic_runner.py``); every other flag
(``--print-timeout``, ``--new-project``, ``--model``) is appended *after*
the prompt text, matching the one invocation pattern confirmed to work.

``--print-timeout 110m`` replaces agy's own 5-minute default: a real
multi-step coding-agent run (repository inspection, code generation,
execution, debugging, validation/repair) routinely takes longer than 5
minutes between model responses, and the default timeout is what actually
killed the original C5 automated attempt (see
outputs/condition_proposals/live/runs/C05/diagnostics/).

``--new-project`` is required for a brand-new working directory (a freshly
built C5-C8 sanitized workspace has never been seen by agy before): probed
live against the installed CLI, without it `agy --print` silently falls
back to reading/writing its own internal
`~/.gemini/antigravity-cli/scratch/` instead of the invoking `cwd`, even
when that `cwd` is already listed in `trustedWorkspaces`. Trust only
exempts a *recognized* project from the approval wall; it does not by
itself make the CLI adopt an unrecognized directory as the active project.
Each automated invocation is a fresh single-prompt session with no
multi-turn continuation to preserve, so treating every run as a new project
is correct, not just a workaround.

``--model`` is appended at execution time (see harnesses/agentic_runner.py)
once resolved to the effort-suffixed slug ``agy models`` actually accepts —
see harnesses/antigravity_support.py. No ``--dangerously-skip-permissions``:
see that module's docstring for the (non-bypass) workspace-trust mechanism
used instead.
"""

from __future__ import annotations

from geo_strategist.harnesses.adapters import HarnessAdapter

# Flags appended AFTER the prompt text (see module docstring for why order
# matters here specifically). Kept separate from command_template, which
# only carries what precedes the prompt.
POST_PROMPT_FLAGS: tuple[str, ...] = ("--print-timeout", "110m", "--new-project")

ADAPTER = HarnessAdapter(
    harness="antigravity",
    command="agy --print <launcher prompt> --print-timeout 110m --new-project",
    skill_source=".agents/skills",
    automation_supported=True,
    command_template=("agy", "--print"),
    prompt_mode="argument",
    timeout_seconds=7200,
)
