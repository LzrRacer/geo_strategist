"""Sandboxed execution of LLM/agent-generated experiment code.

Adapted from AI Scientist-v2's ``treesearch/interpreter.py`` (see
``references/local/ai_scientist/ai_scientist.txt``): a static safety guard
scans generated code before an isolated subprocess with a scrubbed
environment executes it. Used by every condition that runs generated code
(C13-C14 and the manual-harness conditions).
"""

from geo_strategist.agent.codeexec.interpreter import ExecutionResult, Interpreter
from geo_strategist.agent.codeexec.sandbox_guard import (
    audit_artifact_locations,
    scan_generated_code,
)

__all__ = [
    "ExecutionResult",
    "Interpreter",
    "audit_artifact_locations",
    "scan_generated_code",
]
