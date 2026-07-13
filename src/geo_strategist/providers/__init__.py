"""Live LLM provider clients (Gemini, OpenCode Go, OpenRouter) and preflight.

Every client returns a :class:`geo_strategist.providers.base.ChatResult` and
records usage into a shared :class:`geo_strategist.providers.base.CallLedger`
so conditions can publish model-call summaries. No module in this package
ever logs, prints, or persists an API key.
"""

from geo_strategist.providers.base import CallLedger, ChatResult

__all__ = ["CallLedger", "ChatResult"]
