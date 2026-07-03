"""LLM evaluation — how do you score a model's output when there's no single
right answer?

Four complementary angles: surface-overlap text metrics (BLEU/ROUGE/F1/EM),
intrinsic language-model quality (perplexity), whether a model's stated
confidence can be trusted (calibration/ECE), and whether a generated answer
stays grounded in its source (a hallucination/faithfulness heuristic — see
:mod:`optimumai.evaluation.hallucination` for why that last one is an
educational proxy, not a solved problem).
"""

from optimumai.evaluation.calibration import ece, ece_trace
from optimumai.evaluation.hallucination import (
    faithfulness_score,
    faithfulness_trace,
    unsupported_spans,
)
from optimumai.evaluation.perplexity import perplexity, perplexity_trace
from optimumai.evaluation.text_metrics import (
    bleu,
    bleu_trace,
    exact_match,
    rouge_l,
    rouge_l_trace,
    rouge_n,
    rouge_n_trace,
    token_f1,
    token_f1_trace,
)

__all__ = [
    "bleu",
    "bleu_trace",
    "ece",
    "ece_trace",
    "exact_match",
    "faithfulness_score",
    "faithfulness_trace",
    "perplexity",
    "perplexity_trace",
    "rouge_l",
    "rouge_l_trace",
    "rouge_n",
    "rouge_n_trace",
    "token_f1",
    "token_f1_trace",
    "unsupported_spans",
]
