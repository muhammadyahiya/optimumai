"""Real token generation via local Ollama, Hugging Face, Anthropic, or a toy model."""

from optimumai.llm.generate import available_providers, generate, generate_trace

__all__ = ["available_providers", "generate", "generate_trace"]
