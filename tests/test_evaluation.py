import math

import pytest

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

CANDIDATE = "the fast brown fox jumps over the lazy dog"
REFERENCE = "a quick brown fox jumps over the lazy dog"


# --- BLEU -----------------------------------------------------------------
def test_bleu_matches_hand_computed_reference():
    # Independently hand-computed: clipped 1..4-gram precisions are
    # 7/9, 6/8, 5/7, 4/6; both sentences have length 9 so BP = 1.
    score = bleu(CANDIDATE, REFERENCE)
    precisions = [7 / 9, 6 / 8, 5 / 7, 4 / 6]
    expected = math.exp(sum(math.log(p) for p in precisions) / 4)
    assert score == pytest.approx(expected, rel=1e-9)
    assert score == pytest.approx(0.7259795291154771, rel=1e-6)


def test_bleu_identical_strings_score_one():
    assert bleu("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_bleu_zero_ngram_overlap_at_high_order_zeroes_score():
    # Short, no shared 4-grams anywhere -> geometric mean collapses to 0.
    assert bleu("completely different words entirely", "the cat sat on the mat") == 0.0


def test_bleu_brevity_penalty_punishes_short_candidate():
    reference = "the cat sat on the mat today"
    short_but_exact_subset = "the cat"
    score = bleu(short_but_exact_subset, reference, max_n=1)
    # unigram precision is 1.0 (both words appear in reference) but BP < 1 must pull it down
    assert score < 1.0


def test_bleu_trace_shape_and_fields():
    t = bleu_trace(CANDIDATE, REFERENCE)
    # tokenize + 4 precisions + brevity penalty + geometric mean + final score = 8
    assert len(t) == 8
    assert t.result == pytest.approx(bleu(CANDIDATE, REFERENCE))
    assert t.formula
    assert len(t.why_ai) >= 1
    assert t.meta["max_n"] == 4


def test_bleu_rejects_empty_strings():
    with pytest.raises(ValueError):
        bleu_trace("", "something")
    with pytest.raises(ValueError):
        bleu_trace("something", "")


# --- ROUGE-N ----------------------------------------------------------------
def test_rouge_1_matches_hand_computed_reference():
    # Independently hand-computed: 7 shared unigrams out of 9 in each string.
    assert rouge_n(CANDIDATE, REFERENCE, n=1) == pytest.approx(7 / 9)


def test_rouge_2_matches_hand_computed_reference():
    # Independently hand-computed: 6 shared bigrams out of 8 in each string.
    assert rouge_n(CANDIDATE, REFERENCE, n=2) == pytest.approx(6 / 8)


def test_rouge_n_is_recall_oriented():
    # A candidate that repeats the whole reference plus extra words has recall 1.0.
    assert rouge_n("the cat sat on the mat extra words here", "the cat sat on the mat", n=1) == (
        pytest.approx(1.0)
    )


def test_rouge_n_trace_shape():
    t = rouge_n_trace(CANDIDATE, REFERENCE, n=1)
    assert len(t) == 5
    assert t.result == pytest.approx(7 / 9)
    assert "recall" in t.formula.lower()


def test_rouge_n_rejects_reference_shorter_than_n():
    with pytest.raises(ValueError):
        rouge_n_trace("the cat sat", "the", n=2)


# --- ROUGE-L ------------------------------------------------------------------
def test_rouge_l_matches_hand_computed_lcs():
    # Independently hand-computed LCS("the fast brown fox jumps over the lazy dog",
    # "a quick brown fox jumps over the lazy dog") = 7 tokens
    # ("brown fox jumps over the lazy dog").
    t = rouge_l_trace(CANDIDATE, REFERENCE)
    assert t.meta["lcs_length"] == 7
    recall = 7 / 9
    precision = 7 / 9
    expected_f = 2 * precision * recall / (precision + recall)
    assert rouge_l(CANDIDATE, REFERENCE) == pytest.approx(expected_f)


def test_rouge_l_identical_strings_score_one():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_rouge_l_tolerates_reordering_better_than_bigram_overlap():
    # Same bag of words, different order: ROUGE-L still finds a decent LCS,
    # unlike a naive contiguous 4-gram match which would be zero.
    candidate = "cats and dogs are friends"
    reference = "dogs and cats are friends"
    assert rouge_l(candidate, reference) > 0.5


# --- exact match / token F1 ----------------------------------------------------
def test_exact_match_true_and_false():
    assert exact_match("Paris, France", "paris france") == 1.0
    assert exact_match("Paris", "France") == 0.0


def test_token_f1_matches_hand_computed_reference():
    # candidate/reference share {paris}; candidate has 1 extra token.
    t = token_f1_trace("Paris city", "Paris")
    precision = 1 / 2
    recall = 1 / 1
    expected = 2 * precision * recall / (precision + recall)
    assert t.result == pytest.approx(expected)
    assert token_f1("Paris city", "Paris") == pytest.approx(expected)


def test_token_f1_is_order_insensitive():
    assert token_f1("Paris France", "France Paris") == pytest.approx(1.0)


def test_token_f1_trace_has_exact_match_step():
    t = token_f1_trace("same text", "same text")
    assert t.meta["exact_match"] == 1.0
    assert t.result == pytest.approx(1.0)


# --- perplexity -----------------------------------------------------------------
def test_perplexity_matches_cross_entropy_reference():
    probs = [0.5, 0.25, 0.8, 0.4]
    neg_logs = [-math.log(p) for p in probs]
    expected_h = sum(neg_logs) / len(neg_logs)
    expected_ppl = math.exp(expected_h)
    assert perplexity(probs) == pytest.approx(expected_ppl)
    assert perplexity(probs) == pytest.approx(math.sqrt(5), rel=1e-9)


def test_perplexity_of_uniform_guess_equals_branching_factor():
    # Guessing uniformly among k options every time gives perplexity == k exactly.
    k = 8
    probs = [1.0 / k] * 10
    assert perplexity(probs) == pytest.approx(k)


def test_perplexity_confident_correct_beats_unsure():
    confident = perplexity([0.95, 0.9, 0.92])
    unsure = perplexity([0.3, 0.4, 0.35])
    assert confident < unsure


def test_perplexity_trace_shape_and_result():
    t = perplexity_trace([0.5, 0.25, 0.8, 0.4])
    # surprisal, cross-entropy, perplexity = 3 steps
    assert len(t) == 3
    assert t.result == pytest.approx(math.sqrt(5), rel=1e-9)
    assert t.meta["n_tokens"] == 4


def test_perplexity_rejects_probability_not_in_range():
    with pytest.raises(ValueError):
        perplexity_trace([0.5, 0.0])  # 0 gives -inf surprisal, disallowed
    with pytest.raises(ValueError):
        perplexity_trace([0.5, 1.5])  # not a probability
    with pytest.raises(ValueError):
        perplexity_trace([])


# --- calibration / ECE -----------------------------------------------------------
def test_ece_matches_hand_computed_reference():
    # Independently hand-computed on 5 equal-width bins (see module demo values).
    confidences = [0.9, 0.85, 0.95, 0.9, 0.65, 0.7, 0.6, 0.3, 0.4, 0.35]
    correct = [True, True, False, True, True, False, True, False, False, True]
    assert ece(confidences, correct, n_bins=5) == pytest.approx(0.14, rel=1e-9)


def test_ece_is_zero_when_perfectly_calibrated():
    # Every bin's mean confidence exactly equals its accuracy.
    confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    correct = [True] * 9 + [False]  # accuracy 0.9, mean confidence 0.9
    assert ece(confidences, correct, n_bins=5) == pytest.approx(0.0, abs=1e-9)


def test_ece_trace_bin_count_and_weighting():
    # Independently hand-computed: bin [0.0,0.2) has mean_conf=0.1, accuracy=0.0 (gap 0.1);
    # bin [0.8,1.0) has mean_conf=0.9, accuracy=1.0 (gap 0.1); ECE = 0.5*0.1 + 0.5*0.1 = 0.1.
    t = ece_trace([0.9, 0.9, 0.1, 0.1], [True, True, False, False], n_bins=5)
    assert t.meta["n_bins"] == 5
    assert t.result == pytest.approx(0.1, rel=1e-9)
    assert len(t.meta["bins"]) == 2  # only 2 of 5 bins are populated


def test_ece_rejects_mismatched_lengths_and_bad_confidence():
    with pytest.raises(ValueError):
        ece_trace([0.5, 0.6], [True])
    with pytest.raises(ValueError):
        ece_trace([1.5], [True])
    with pytest.raises(ValueError):
        ece_trace([], [])


# --- hallucination / faithfulness heuristic --------------------------------------
CONTEXT = "The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris."


def test_faithfulness_fully_grounded_answer_scores_one():
    grounded = "The Eiffel Tower was completed in 1889 and stands 330 meters tall."
    assert faithfulness_score(grounded, CONTEXT) == pytest.approx(1.0)


def test_faithfulness_partially_hallucinated_answer_matches_hand_computed_reference():
    # Independently hand-computed: answer content tokens are
    # {eiffel, tower, completed, 1889, designed, leonardo, da, vinci} = 8 tokens,
    # of which {eiffel, tower, completed, 1889} = 4 are in the context's vocabulary.
    hallucinated = "The Eiffel Tower was completed in 1889 and was designed by Leonardo da Vinci."
    assert faithfulness_score(hallucinated, CONTEXT) == pytest.approx(0.5)


def test_faithfulness_flags_the_right_unsupported_spans():
    hallucinated = "The Eiffel Tower was completed in 1889 and was designed by Leonardo da Vinci."
    flagged = unsupported_spans(hallucinated, CONTEXT)
    assert set(flagged) == {"designed", "leonardo", "da", "vinci"}


def test_faithfulness_trace_shape_and_result():
    hallucinated = "The Eiffel Tower was completed in 1889 and was designed by Leonardo da Vinci."
    t = faithfulness_trace(hallucinated, CONTEXT)
    assert t.result == pytest.approx(0.5)
    assert "heuristic" in t.formula.lower()
    assert t.meta["supported"] and t.meta["unsupported"]
    assert len(t.why_ai) >= 1


def test_faithfulness_rejects_empty_answer_or_context():
    with pytest.raises(ValueError):
        faithfulness_trace("", CONTEXT)
    with pytest.raises(ValueError):
        faithfulness_trace("some answer", "")


# --- demo() smoke tests (deterministic, fast) ------------------------------------
def test_all_demos_run_and_produce_a_trace_with_a_result():
    from optimumai.evaluation.calibration import demo as calibration_demo
    from optimumai.evaluation.hallucination import demo as hallucination_demo
    from optimumai.evaluation.perplexity import demo as perplexity_demo
    from optimumai.evaluation.text_metrics import demo as text_metrics_demo

    for demo_fn in (text_metrics_demo, perplexity_demo, calibration_demo, hallucination_demo):
        t = demo_fn()
        assert t.result is not None
        assert len(t) > 0
        assert t.why_ai
