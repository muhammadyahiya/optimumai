"""Optional LLM tutor layer for OptimumAI.

The core math of OptimumAI runs fully offline and never touches this module.
The :class:`Tutor` here is a *thin, optional* wrapper around `litellm` that adds
natural-language explanations on top of the deterministic traces.

It is intentionally forgiving: if the ``[llm]`` extra is not installed (no
``litellm``) or no API key is configured, every method degrades to a friendly,
actionable message instead of raising. Install with::

    pip install "optimumai[llm]"

and set ``OPTIMUMAI_API_KEY`` (plus optionally ``OPTIMUMAI_LLM_MODEL`` /
``OPTIMUMAI_API_BASE``) to enable it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from optimumai.config import get_settings
from optimumai.core.explain import ExplainLevel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from optimumai.core.trace import Trace

_OFFLINE_MESSAGE = (
    "The OptimumAI LLM tutor is not available right now, so here is what is "
    "missing:\n"
    '  1. Install the optional dependency:  pip install "optimumai[llm]"\n'
    "  2. Configure a key:  export OPTIMUMAI_API_KEY=sk-...\n"
    "     (optionally OPTIMUMAI_LLM_MODEL and OPTIMUMAI_API_BASE)\n\n"
    "None of this is required for the core math — every operation, trace, and "
    "explanation works fully offline without an LLM."
)

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
def _litellm_available() -> bool:
    """Return ``True`` if ``litellm`` can be imported (result is cached)."""
    try:
        import litellm  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


class Tutor:
    """A friendly, optional LLM tutor for the math behind AI.

    The tutor is safe to construct and call unconditionally: when the optional
    ``litellm`` dependency or an API key is absent, its methods return a clear
    offline message rather than raising.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self.model = model if model is not None else settings.llm_model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )

    @property
    def available(self) -> bool:
        """True only when ``litellm`` is importable *and* an API key is set."""
        return _litellm_available() and self._settings.api_key is not None

    def ask(self, question: str, system: str | None = None) -> str:
        """Ask the tutor a free-form question and return its answer.

        Returns a helpful offline message (never raises) when the tutor is
        unavailable, and a readable error string if the LLM call itself fails.
        """
        if not self.available:
            return _OFFLINE_MESSAGE

        try:
            import litellm

            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": question})

            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                api_key=self._settings.api_key,
                api_base=self._settings.api_base,
            )
            return self._extract_text(response)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            return f"The LLM tutor hit an error and could not answer: {exc}"

    def explain(
        self,
        concept: str,
        level: ExplainLevel | str = ExplainLevel.INTERMEDIATE,
    ) -> str:
        """Explain a math/AI ``concept`` at the requested detail ``level``."""
        parsed = ExplainLevel.parse(level)
        system = (
            "You are a world-class AI tutor who explains the math behind AI "
            "from first principles. You build intuition before formalism, you "
            "are precise, and you never invent facts.\n"
            f"{_LEVEL_GUIDANCE[parsed]}"
        )
        question = (
            f"Explain the following concept: {concept}\n\n"
            f"Tailor the depth to a '{parsed.value}' audience."
        )
        return self.ask(question, system=system)

    def explain_trace(
        self,
        trace: Trace,
        level: ExplainLevel | str = ExplainLevel.INTERMEDIATE,
    ) -> str:
        """Add LLM intuition on top of an OptimumAI :class:`Trace`.

        Summarizes the trace's operation, formula, and steps into a prompt and
        asks the tutor for extra intuition. Returns the offline message (never
        raises) when the tutor is unavailable.
        """
        if not self.available:
            return _OFFLINE_MESSAGE

        parsed = ExplainLevel.parse(level)
        system = (
            "You are a world-class AI tutor who explains the math behind AI "
            "from first principles. You are given a deterministic computation "
            "trace and must add intuition and context — never contradict the "
            "numbers you are shown.\n"
            f"{_LEVEL_GUIDANCE[parsed]}"
        )
        question = self._summarize_trace(trace)
        return self.ask(question, system=system)

    @staticmethod
    def _summarize_trace(trace: Trace) -> str:
        """Render a :class:`Trace` into a compact prompt for the LLM."""
        lines = [f"Operation: {trace.op}"]
        if trace.formula:
            lines.append(f"Formula: {trace.formula}")
        if trace.complexity:
            lines.append(f"Complexity: {trace.complexity}")
        if trace.steps:
            lines.append("Steps:")
            for step in trace.steps:
                lines.append(f"  {step.index}. {step.title}: {step.expression}")
        if trace.result is not None:
            lines.append(f"Result: {trace.result}")
        if trace.why_ai:
            lines.append("Where this shows up in AI:")
            lines.extend(f"  - {point}" for point in trace.why_ai)

        summary = "\n".join(lines)
        return (
            "Here is a computation trace from the OptimumAI SDK:\n\n"
            f"{summary}\n\n"
            "Give extra intuition about what is happening and why it matters, "
            "without repeating every number verbatim."
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        """Pull the assistant text out of a litellm/OpenAI-style response."""
        try:
            return response.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - provider-shape dependent
            return str(response)
