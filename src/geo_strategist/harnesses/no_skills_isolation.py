"""Workspace isolation for the C5-C8 no-Skills coding-agent controls.

A launcher prompt telling the agent "don't read the Skills packages" is an
honor-system instruction, not a contamination control: the automated
(non-interactive) harness runs with the repository as its working directory,
where `.agents/skills/` and `.claude/skills/` are physically present and
readable regardless of what the prompt says. This module makes the control
enforceable instead: it builds a sanitized workspace — a directory of
symlinks mirroring the real repository, minus the Skill package directories
and with a Skills-reference-free `AGENTS.md` — and that sanitized directory,
not the real repo root, becomes the automated harness's working directory.

Symlinks (not a full copy) keep this cheap even though the repository
includes large data directories: only the top-level directory entries are
symlinked, so no file content is duplicated.
"""

from __future__ import annotations

import re
from pathlib import Path

# Never symlinked into the sanitized workspace. `.claude/skills` also holds
# only Skill packages (as symlinks back into `.agents/skills`); everything
# else under `.claude/` (if anything) is not part of the Skills contract, so
# only the `skills` subdirectory is withheld, not the whole `.claude/` tree.
#
# `.git`, `.codex`, `.pytest_cache`, and `.runs` are withheld too: CLI
# sandboxes (observed with Codex's workspace-write/bubblewrap sandbox) refuse
# to enforce their read-only/writable path policy when a directory symlink
# crosses into a tool's own state/cache directory, failing every write in
# the session ("cannot enforce sandbox read-only path ... because it crosses
# writable symlink"). None of these are needed by the no-Skills coding-agent
# task (which needs the working tree and real project data, not VCS history,
# other CLIs' own config state, test caches, or this project's own
# experiment-run bookkeeping) — withholding them costs nothing functionally
# and sidesteps the sandbox conflict regardless of which directory a given
# CLI's sandbox happens to be sensitive about. `.scratch` is withheld for a
# related reason: it is where this module builds its own sanitized
# workspaces (see below), so symlinking it back into a workspace would
# create a self-referential loop back to one of that workspace's own
# ancestor directories.
_WITHHELD_TOP_LEVEL: frozenset[str] = frozenset(
    {".agents", ".git", ".codex", ".pytest_cache", ".runs", ".scratch"})
_WITHHELD_NESTED: dict[str, frozenset[str]] = {".claude": frozenset({"skills"})}
_REWRITTEN_TOP_LEVEL: frozenset[str] = frozenset({"AGENTS.md"})

# Lines/sections of AGENTS.md that name the Skills contract; stripped from
# the sanitized copy so a no-Skills session cannot learn the contract exists
# just by reading AGENTS.md, even though it never has filesystem access to
# the packages themselves.
_SKILLS_REFERENCE_RE = re.compile(
    r"skill|\.agents/skills|\.claude/skills|agents_md_skills", re.IGNORECASE)


def _sanitized_agents_md(text: str) -> str:
    kept_lines = [line for line in text.splitlines() if not _SKILLS_REFERENCE_RE.search(line)]
    return "\n".join(kept_lines) + "\n"


def prepare_no_skills_workspace(repo_root: Path, run_dir: Path) -> Path:
    """Build (or reuse) a sanitized symlink workspace for one C5-C8 run.

    Returns the sanitized workspace path; the automated harness adapter must
    be invoked with this as its working directory instead of ``repo_root``.

    The workspace is built *outside* ``run_dir`` (under ``repo_root/.scratch``)
    rather than as a child of it. ``run_dir`` normally lives under the
    project's own ``outputs/`` tree, which this workspace also symlinks back
    to (so the agent can read/write project outputs) — nesting the workspace
    inside ``run_dir`` would make that symlink resolve back to one of the
    workspace's own ancestor directories, a self-referential loop that a
    CLI's sandbox (observed with Codex's bubblewrap-based sandbox) correctly
    refuses to treat as writable.
    """

    workspace = repo_root / ".scratch" / "no_skills_workspaces" / run_dir.name
    if workspace.exists():
        return workspace
    workspace.mkdir(parents=True, exist_ok=True)
    for entry in repo_root.iterdir():
        if entry.name in _WITHHELD_TOP_LEVEL:
            continue
        if entry.name in _REWRITTEN_TOP_LEVEL:
            continue
        if entry.name in _WITHHELD_NESTED:
            _symlink_dir_minus(entry, workspace / entry.name, _WITHHELD_NESTED[entry.name])
            continue
        (workspace / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    agents_md = repo_root / "AGENTS.md"
    if agents_md.exists():
        (workspace / "AGENTS.md").write_text(
            _sanitized_agents_md(agents_md.read_text(encoding="utf-8")), encoding="utf-8")
    return workspace


def _symlink_dir_minus(source_dir: Path, dest_dir: Path, withheld_names: frozenset[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in source_dir.iterdir():
        if entry.name in withheld_names:
            continue
        (dest_dir / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())


def skills_dirs_present(workspace: Path) -> bool:
    """True if any Skill package path is reachable from ``workspace`` — used
    to record (and fail loudly on) an isolation regression, never silently."""

    return (workspace / ".agents" / "skills").exists() or (workspace / ".claude" / "skills").exists()
