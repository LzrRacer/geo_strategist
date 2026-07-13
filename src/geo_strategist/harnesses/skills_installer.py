"""Validation helpers for repository-local Skill package compatibility."""

from __future__ import annotations

from pathlib import Path

from geo_strategist.agent.skill_registry import validate_skill_packages


def validate_installed_skill_packages(repo_root: Path) -> list[str]:
    """Return package installation issues for .agents and .claude skills."""

    return validate_skill_packages(repo_root)
