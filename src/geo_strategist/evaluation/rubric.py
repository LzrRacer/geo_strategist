"""Evaluation rubric loader for scaffold validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Criterion(BaseModel):
    """A weighted evaluation criterion."""

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(gt=0)
    description: str = Field(min_length=1)


class ScorerType(str, Enum):
    """Supported score sources."""

    HUMAN_REVIEWER = "human_reviewer"
    LLM_AS_JUDGE = "llm_as_judge"
    DETERMINISTIC_VALIDATOR = "deterministic_validator"


class Score(BaseModel):
    """One rubric score from a human, LLM judge, or deterministic validator."""

    model_config = ConfigDict(extra="forbid")

    score_id: str = Field(min_length=1)
    proposal_id: str | None = None
    scorer_type: ScorerType
    criterion: str = Field(min_length=1)
    rubric_version: int = Field(default=1, ge=1)
    value: float
    scale_min: float
    scale_max: float
    rationale: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _score_must_be_within_scale(self) -> "Score":
        if self.scale_min > self.scale_max:
            raise ValueError("score scale_min cannot exceed scale_max")
        if not self.scale_min <= self.value <= self.scale_max:
            raise ValueError("score value must be within the configured range")
        return self


class Rubric(BaseModel):
    """Unified rubric for all experiment conditions."""

    model_config = ConfigDict(extra="forbid")

    version: int
    applies_to_conditions: list[str] = Field(default_factory=list)
    scale: dict[Any, Any]
    criteria: dict[str, Criterion]
    hard_fail_checks: list[str] = Field(default_factory=list)

    def total_weight(self) -> float:
        """Return the sum of all criterion weights."""

        return sum(criterion.weight for criterion in self.criteria.values())

    @property
    def scale_min(self) -> float:
        """Configured minimum score."""

        return float(self.scale["min"])

    @property
    def scale_max(self) -> float:
        """Configured maximum score."""

        return float(self.scale["max"])

    @model_validator(mode="after")
    def _validate_total_weight(self) -> "Rubric":
        if abs(self.total_weight() - 1.0) > 1e-9:
            raise ValueError("rubric criterion weights must sum to 1.0")
        if float(self.scale["min"]) >= float(self.scale["max"]):
            raise ValueError("rubric scale min must be less than max")
        return self


def load_rubric(path: str | Path = "configs/evaluation_rubric.yaml") -> Rubric:
    """Load and validate a rubric YAML file."""

    rubric_path = Path(path)
    with rubric_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return Rubric.model_validate(raw)
