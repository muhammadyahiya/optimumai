"""Optional LLM tutor layer for OptimumAI.

The core math of OptimumAI runs fully offline and never touches this module.
:class:`Tutor` adds natural-language explanations on top of the deterministic
traces, and it "just works" when a key is present: it auto-detects the Anthropic
SDK + ``ANTHROPIC_API_KEY``, then falls back to ``litellm`` with any provider key
(``OPTIMUMAI_API_KEY`` / ``OPENAI_API_KEY``). If nothing is configured it degrades
to a clear, actionable message instead of raising.

Enable it with::

    pip install "optimumai[llm]"
    export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY / OPTIMUMAI_API_KEY
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from optimumai.config import get_settings
from optimumai.core.explain import ExplainLevel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from optimumai.core.trace import Trace

_DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Depth guidance injected into the system prompt, keyed by explain level.
_LEVEL_GUIDANCE: dict[ExplainLevel, str] = {
    ExplainLevel.BEGINNER: (
        "Explain for a curious beginner. Avoid jargon, use plain language and "
        "everyday analogies, and keep the math to a minimum."
    ),
    ExplainLevel.INTERMEDIATE: (
        "Explain for someone comfortable with high-school math. Introduce the "
        "key formulas, define symbols, and give clear intuition."
    ),
    ExplainLevel.ENGINEER: (
        "Explain for a working ML engineer. Include the precise formulas, "
        "computational complexity, numerical-stability concerns, and practical "
        "trade-offs."
    ),
    ExplainLevel.RESEARCHER: (
        "Explain for a researcher. Be rigorous about the derivation, discuss "
        "assumptions and edge cases, and connect it to the broader literature."
    ),
}


@lru_cache(maxsize=1)
def _has_module(name: str) -> bool:
    """True if ``name`` can be imported (cached)."""
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _anthropic_key() -> str | None:
    settings = get_settings()
    key = settings.api_key or os.environ.get("ANTHROPIC_API_KEY")
    # An OPTIMUMAI_API_KEY that looks like an Anthropic key also counts.
    if key and (key.startswith("sk-ant-") or os.environ.get("ANTHROPIC_API_KEY")):
        return key if key.startswith("sk-ant-") else os.environ["ANTHROPIC_API_KEY"]
    return os.environ.get("ANTHROPIC_API_KEY")


def _resolve_backend() -> tuple[str, str, str] | None:
    """Pick a usable (backend, model, api_key), or None if nothing is configured.

    Preference order:
      1. Anthropic SDK + an Anthropic key (best experience for Claude Code users)
      2. litellm + any provider key (OPTIMUMAI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
    """
    settings = get_settings()
    override_model = settings.llm_model if settings.llm_model not in (None, "") else None

    ak = _anthropic_key()
    if ak and _has_module("anthropic"):
        model = override_model if (override_model and "claude" in override_model) else \
            _DEFAULT_ANTHROPIC_MODEL
        return ("anthropic", model, ak)

    if _has_module("litellm"):
        if settings.api_key:
            return ("litellm", override_model or _DEFAULT_OPENAI_MODEL, settings.api_key)
        if os.environ.get("ANTHROPIC_API_KEY"):
            model = override_model if (override_model and "/" in override_model) else \
                f"anthropic/{_DEFAULT_ANTHROPIC_MODEL}"
            return ("litellm", model, os.environ["ANTHROPIC_API_KEY"])
        if os.environ.get("OPENAI_API_KEY"):
            model = override_model or _DEFAULT_OPENAI_MODEL
            return ("litellm", model, os.environ["OPENAI_API_KEY"])
    return None


def _offline_message() -> str:
    has_anthropic = _has_module("anthropic")
    has_litellm = _has_module("litellm")
    has_key = bool(
        _anthropic_key()
        or os.environ.get("OPENAI_API_KEY")
        or get_settings().api_key
    )
    lines = ["The OptimumAI tutor isn't active yet. Here's what's missing:"]
    if not (has_anthropic or has_litellm):
        lines.append('  • a client library:  pip install "optimumai[llm]"')
    if not has_key:
        lines.append("  • an API key:  export ANTHROPIC_API_KEY=sk-ant-...")
        lines.append("    (or OPENAI_API_KEY, or OPTIMUMAI_API_KEY)")
    lines.append("")
    lines.append(
        "None of this is required for the core math — every operation, trace, and "
        "explanation works fully offline."
    )
    return "\n".join(lines)


class Tutor:
    """A friendly, optional LLM tutor for the math behind AI.

    Safe to construct and call unconditionally: when no client library or key is
    available, methods return a clear message rather than raising.
    """

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        settings = get_settings()
        self._model_override = model
        self.temperature = temperature if temperature is not None else settings.temperature

    @property
    def available(self) -> bool:
        """True when a client library *and* an API key are both available."""
        return _resolve_backend() is not None

    def ask(self, question: str, system: str | None = None) -> str:
        """Ask a free-form question. Never raises; returns guidance if unavailable."""
        backend = _resolve_backend()
        if backend is None:
            return _offline_message()
        provider, model, api_key = backend
        model = self._model_override or model
        try:
            if provider == "anthropic":
                return self._ask_anthropic(question, system, model, api_key)
            return self._ask_litellm(question, system, model, api_key)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            return f"The tutor hit an error and could not answer: {exc}"

    def explain(self, concept: str, level: ExplainLevel | str = ExplainLevel.INTERMEDIATE) -> str:
        """Explain a math/AI ``concept`` at the requested detail ``level``."""
        parsed = ExplainLevel.parse(level)
        system = (
            "You are a world-class AI tutor who explains the math behind AI from "
            "first principles. You build intuition before formalism, you are "
            f"precise, and you never invent facts.\n{_LEVEL_GUIDANCE[parsed]}"
        )
        return self.ask(f"Explain this concept: {concept}", system=system)

    def explain_trace(
        self, trace: Trace, level: ExplainLevel | str = ExplainLevel.INTERMEDIATE
    ) -> str:
        """Add LLM intuition on top of an OptimumAI :class:`Trace`."""
        parsed = ExplainLevel.parse(level)
        system = (
            "You are a world-class AI tutor. You are given a deterministic "
            "computation trace and must add intuition and context — never "
            f"contradict the numbers shown.\n{_LEVEL_GUIDANCE[parsed]}"
        )
        return self.ask(self._summarize_trace(trace), system=system)

    # ------------------------------------------------------------------ backends
    def _ask_anthropic(self, question: str, system: str | None, model: str, key: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system or "You are a helpful AI-math tutor.",
            messages=[{"role": "user", "content": question}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    def _ask_litellm(self, question: str, system: str | None, model: str, key: str) -> str:
        import litellm

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        resp = litellm.completion(
            model=model, messages=messages, temperature=self.temperature, api_key=key
        )
        try:
            return resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - provider-shape dependent
            return str(resp)

    @staticmethod
    def _summarize_trace(trace: Trace) -> str:
        lines = [f"Operation: {trace.op}"]
        if trace.formula:
            lines.append(f"Formula: {trace.formula}")
        if trace.steps:
            lines.append("Steps:")
            lines.extend(f"  {s.index}. {s.title}: {s.expression}" for s in trace.steps)
        if trace.result is not None:
            lines.append(f"Result: {trace.result}")
        if trace.why_ai:
            lines.append("Where this shows up in AI:")
            lines.extend(f"  - {point}" for point in trace.why_ai)
        return (
            "Here is a computation trace from OptimumAI:\n\n"
            + "\n".join(lines)
            + "\n\nGive extra intuition about what is happening and why it matters."
        )
