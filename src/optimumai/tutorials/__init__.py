"""Runnable, explained tutorials for the tools you actually use.

    from optimumai.tutorials import get_tutorial, list_tutorials
    get_tutorial("numpy").run()

Topics: ``numpy`` (runs on the base install), ``matplotlib`` (needs ``[viz]``),
``pytorch`` (built on OptimumAI's own engine so it runs torch-free, with the real
torch code shown), and ``finetuning`` (a numpy toy SFT → LoRA → DPO pipeline plus
the production HF/PEFT/TRL equivalents).
"""

from __future__ import annotations

from optimumai.tutorials.core import Cell, Tutorial, get_tutorial, list_tutorials

__all__ = ["Cell", "Tutorial", "get_tutorial", "list_tutorials"]
