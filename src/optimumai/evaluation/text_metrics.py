"""Surface-overlap text metrics — BLEU, ROUGE, exact match, token F1.

These are the oldest and still most common way to score a generated string
against one or more reference strings: chop both into tokens/n-grams and count
how much they overlap.

* **BLEU** (BiLingual Evaluation Understudy) — precision-oriented. For each
  n-gram order it computes a *clipped* precision (a candidate n-gram can only
  count as a match up to the number of times it appears in the reference, so
  repeating a common word cannot inflate the score), then combines orders 1..N
  with a geometric mean and multiplies by a brevity penalty that punishes
  short, low-recall candidates the precision alone would miss.
* **ROUGE-N** — the recall-oriented mirror of BLEU: what fraction of the
  reference's n-grams did the candidate reproduce? ROUGE-L instead measures
  the Longest Common Subsequence (LCS), which tolerates gaps/reordering that
  contiguous n-grams cannot.
* **Exact match / token F1** — the extractive-QA staples (SQuAD-style):
  exact match is a strict string-equality check after normalization; F1
  treats both strings as unordered bags of tokens and computes precision,
  recall, and their harmonic mean.

What these metrics do NOT capture
----------------------------------
All four are *surface-overlap* metrics: they compare tokens, not meaning.
"The cat sat on the mat" and "A feline rested upon the rug" describe the same
scene but share almost no n-grams — a fluent paraphrase, a correct
translation using different words, or a semantically equivalent answer with
different phrasing will all score poorly. They also can't detect fluency,
factual correctness, or logical consistency; a candidate can shuffle the
reference's words into nonsense and still score well on unigram overlap.
Modern practice increasingly pairs (or replaces) these with embedding-based
metrics (BERTScore, MoverScore) or LLM-as-judge scoring — see
:mod:`optimumai.evaluation.hallucination` for a related, more candid
discussion of what "correctness" even means for open-ended generation.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from optimumai.core._fmt import num
from optimumai.core.trace import Trace


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization (good enough to teach with)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _clipped_precision(candidate: list[str], reference: list[str], n: int) -> tuple[int, int]:
    """Return ``(clipped_matches, total_candidate_ngrams)`` for order ``n``."""
    cand_counts = Counter(_ngrams(candidate, n))
    ref_counts = Counter(_ngrams(reference, n))
    clipped = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    total = sum(cand_counts.values())
    return clipped, total


def _validate_nonempty(text: str, name: str) -> None:
    if not text or not text.strip():
        raise ValueError(f"{name} must be a non-empty string")


# --------------------------------------------------------------------------- BLEU
def bleu_trace(candidate: str, reference: str, max_n: int = 4) -> Trace:
    """Build the full BLEU trace: per-order clipped precision, then the brevity penalty."""
    _validate_nonempty(candidate, "candidate")
    _validate_nonempty(reference, "reference")
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")

    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        raise ValueError("candidate/reference must contain at least one alphanumeric token")

    t = Trace(
        op="bleu",
        formula="BLEU = BP · exp((1/N) Σₙ log pₙ),  "
        "pₙ = clipped n-gram matches / candidate n-grams",
        complexity=f"O(L) per n-gram order, O(N·L) total for N={max_n}",
        why_ai=[
            "The classic machine-translation metric; still a default sanity check",
            "Clipping matches to the reference count prevents 'the the the the' from "
            "scoring perfectly by repeating a common word",
            "The brevity penalty stops short, safe candidates from gaming precision "
            "(a 1-word candidate that matches can have precision 1.0 but says nothing)",
        ],
        meta={"candidate": candidate, "reference": reference, "max_n": max_n},
    )
    t.add(
        "Tokenize",
        f"candidate: {cand_tokens}\nreference: {ref_tokens}",
        detail=f"{len(cand_tokens)} candidate tokens, {len(ref_tokens)} reference tokens.",
    )

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        clipped, total = _clipped_precision(cand_tokens, ref_tokens, n)
        p_n = clipped / total if total else 0.0
        precisions.append(p_n)
        t.add(
            f"{n}-gram precision (clipped)",
            f"p_{n} = {clipped} clipped matches / {total} candidate {n}-grams = {num(p_n)}",
            p_n,
            detail="A match counts at most as many times as it appears in the reference.",
        )

    c_len, r_len = len(cand_tokens), len(ref_tokens)
    brevity_penalty = 1.0 if c_len > r_len else math.exp(1 - r_len / c_len)
    t.add(
        "Brevity penalty",
        f"BP = 1 if c_len > r_len else e^(1 − r_len/c_len);  c_len={c_len}, r_len={r_len} "
        f"→ BP = {num(brevity_penalty)}",
        brevity_penalty,
        detail="Punishes candidates that are shorter than the reference (precision alone "
        "rewards saying less).",
    )

    if min(precisions) <= 0.0:
        geo_mean = 0.0
        t.add(
            "Geometric mean of precisions",
            "at least one n-gram order has zero matches → geometric mean = 0",
            geo_mean,
            detail="This is BLEU's well-known brittleness at the sentence level: a single "
            "missing 4-gram zeroes the whole score.",
        )
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        t.add(
            "Geometric mean of precisions",
            f"exp((1/{max_n}) Σ log pₙ) = {num(geo_mean)}",
            geo_mean,
        )

    score = brevity_penalty * geo_mean
    t.add("BLEU score", f"BP × geometric mean = {num(brevity_penalty)} × {num(geo_mean)} "
          f"= {num(score)}", score)
    t.result = score
    return t


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Return the BLEU score of ``candidate`` against ``reference`` in ``[0, 1]``."""
    return bleu_trace(candidate, reference, max_n=max_n).result


# -------------------------------------------------------------------------- ROUGE
def rouge_n_trace(candidate: str, reference: str, n: int = 1) -> Trace:
    """Build the ROUGE-N trace: recall/precision/F1 of order-``n`` n-gram overlap."""
    _validate_nonempty(candidate, "candidate")
    _validate_nonempty(reference, "reference")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    cand_grams = Counter(_ngrams(cand_tokens, n))
    ref_grams = Counter(_ngrams(ref_tokens, n))
    if not ref_grams:
        raise ValueError(f"reference has fewer than {n} tokens; cannot form {n}-grams")

    overlap = sum((cand_grams & ref_grams).values())
    n_cand = sum(cand_grams.values())
    n_ref = sum(ref_grams.values())
    recall = overlap / n_ref if n_ref else 0.0
    precision = overlap / n_cand if n_cand else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    t = Trace(
        op="rouge_n",
        formula=f"ROUGE-{n} recall = |overlap {n}-grams| / |reference {n}-grams|",
        complexity=f"O(L) for n={n}",
        why_ai=[
            "The standard summarization metric — recall-first because a good summary "
            "should cover the reference's key n-grams",
            "ROUGE-1/ROUGE-2 (unigram/bigram overlap) are the most commonly reported "
            "orders in papers",
            "Like BLEU, this is surface overlap: a summary that paraphrases every "
            "sentence perfectly can score near zero",
        ],
        meta={
            "candidate": candidate,
            "reference": reference,
            "n": n,
            "overlap": overlap,
            "n_candidate_ngrams": n_cand,
            "n_reference_ngrams": n_ref,
        },
    )
    t.add(
        f"Count {n}-grams",
        f"candidate: {n_cand} {n}-grams, reference: {n_ref} {n}-grams",
        detail=f"candidate {n}-grams: {list(cand_grams.elements())}\n"
        f"reference {n}-grams: {list(ref_grams.elements())}",
    )
    t.add(
        "Overlap (multiset intersection)",
        f"|overlap| = {overlap}",
        overlap,
        detail="Counter & Counter clips each shared n-gram to the smaller of the two counts "
        "— the same clipping idea as BLEU.",
    )
    t.add("Recall", f"{overlap} / {n_ref} = {num(recall)}", recall,
          detail="Of the reference's n-grams, how many did the candidate reproduce?")
    t.add("Precision", f"{overlap} / {n_cand} = {num(precision)}", precision,
          detail="Of the candidate's n-grams, how many are actually in the reference?")
    t.add("F1", f"2·P·R / (P+R) = {num(f1)}", f1)
    t.result = recall
    return t


def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    """Return ROUGE-N recall (the conventionally reported value) in ``[0, 1]``."""
    return rouge_n_trace(candidate, reference, n=n).result


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Classic O(len(a)·len(b)) dynamic-program LCS length."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def rouge_l_trace(candidate: str, reference: str) -> Trace:
    """Build the ROUGE-L trace: LCS length → recall/precision/F-measure."""
    _validate_nonempty(candidate, "candidate")
    _validate_nonempty(reference, "reference")

    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        raise ValueError("candidate/reference must contain at least one alphanumeric token")

    lcs = _lcs_length(cand_tokens, ref_tokens)
    recall = lcs / len(ref_tokens)
    precision = lcs / len(cand_tokens)
    f_measure = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    t = Trace(
        op="rouge_l",
        formula="LCS(candidate, reference) → R = LCS/|ref|, P = LCS/|cand|, "
        "F = 2PR/(P+R)",
        complexity=f"O(m·n) DP table, m={len(cand_tokens)}, n={len(ref_tokens)}",
        why_ai=[
            "LCS tolerates word order gaps that contiguous n-grams cannot, so it "
            "rewards 'the model that answered correctly' matching 'the model answered "
            "the question correctly' more fairly than ROUGE-2 would",
            "Used alongside ROUGE-1/2 in nearly every summarization paper",
        ],
        meta={
            "candidate": candidate,
            "reference": reference,
            "lcs_length": lcs,
            "candidate_tokens": len(cand_tokens),
            "reference_tokens": len(ref_tokens),
        },
    )
    t.add(
        "Tokenize",
        f"candidate ({len(cand_tokens)}): {cand_tokens}\nreference ({len(ref_tokens)}): "
        f"{ref_tokens}",
    )
    t.add(
        "Longest Common Subsequence",
        f"LCS length = {lcs}",
        lcs,
        detail="A subsequence need not be contiguous — it can skip words in both strings, "
        "which is what lets ROUGE-L see past small insertions/deletions.",
    )
    t.add("Recall", f"{lcs} / {len(ref_tokens)} = {num(recall)}", recall)
    t.add("Precision", f"{lcs} / {len(cand_tokens)} = {num(precision)}", precision)
    t.add("F-measure", f"2·P·R / (P+R) = {num(f_measure)}", f_measure)
    t.result = f_measure
    return t


def rouge_l(candidate: str, reference: str) -> float:
    """Return the ROUGE-L F-measure in ``[0, 1]``."""
    return rouge_l_trace(candidate, reference).result


# --------------------------------------------------------------- exact match / F1
def exact_match(candidate: str, reference: str) -> float:
    """1.0 if the normalized token sequences are identical, else 0.0 (SQuAD-style)."""
    _validate_nonempty(candidate, "candidate")
    _validate_nonempty(reference, "reference")
    return 1.0 if _tokenize(candidate) == _tokenize(reference) else 0.0


def token_f1_trace(candidate: str, reference: str) -> Trace:
    """Build the token-F1 trace: bag-of-tokens precision/recall/F1 (SQuAD-style)."""
    _validate_nonempty(candidate, "candidate")
    _validate_nonempty(reference, "reference")

    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((cand_counts & ref_counts).values())

    precision = overlap / len(cand_tokens) if cand_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    em = 1.0 if cand_tokens == ref_tokens else 0.0

    t = Trace(
        op="token_f1",
        formula="F1 = 2PR/(P+R),  P = overlap/|candidate|,  R = overlap/|reference|",
        complexity="O(L)",
        why_ai=[
            "The standard extractive-QA metric (SQuAD): exact match is strict, "
            "token F1 gives partial credit for near-miss spans",
            "Order-insensitive — 'Paris, France' and 'France, Paris' score identically, "
            "which is a feature for short-answer QA and a blind spot for anything "
            "order-sensitive",
        ],
        meta={
            "candidate": candidate,
            "reference": reference,
            "overlap": overlap,
            "exact_match": em,
        },
    )
    t.add("Tokenize", f"candidate: {cand_tokens}\nreference: {ref_tokens}")
    t.add("Exact match", f"{cand_tokens} == {ref_tokens} → {bool(em)}", em)
    t.add(
        "Token overlap (bag intersection)",
        f"|overlap| = {overlap}",
        overlap,
        detail="Order-insensitive: only token multiset membership counts.",
    )
    t.add("Precision", f"{overlap} / {len(cand_tokens)} = {num(precision)}", precision)
    t.add("Recall", f"{overlap} / {len(ref_tokens)} = {num(recall)}", recall)
    t.add("F1", f"2·P·R / (P+R) = {num(f1)}", f1)
    t.result = f1
    return t


def token_f1(candidate: str, reference: str) -> float:
    """Return the token-level F1 of ``candidate`` against ``reference`` in ``[0, 1]``."""
    return token_f1_trace(candidate, reference).result


def demo(seed: int = 0) -> Trace:
    """BLEU on a well-matched candidate/reference pair with a clean 4-gram signal."""
    candidate = "the fast brown fox jumps over the lazy dog"
    reference = "a quick brown fox jumps over the lazy dog"
    return bleu_trace(candidate, reference)
