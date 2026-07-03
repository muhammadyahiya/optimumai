import math

import numpy as np
import pytest

from optimumai.nlp.bpe import BPETokenizer, bpe_trace
from optimumai.nlp.edit_distance import edit_distance, edit_distance_trace, edit_script
from optimumai.nlp.ngram import NGramModel, ngram_trace
from optimumai.nlp.tfidf import TfidfVectorizer, tfidf, tfidf_trace
from optimumai.nlp.word2vec import SkipGramModel, skipgram_pairs, word2vec_trace

# --- TF-IDF -----------------------------------------------------------------
DOCS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
]


def test_tfidf_matches_hand_computed_reference():
    # Independently hand-computed with smooth idf = log(N/df) + 1, N=3:
    # df(the)=2 -> idf=log(3/2)+1=1.405465; df(cat)=1, df(mat)=1 -> idf=log(3)+1=2.098612
    # tf('the', doc0) = 2/6 = 0.333333; tf('cat', doc0) = tf('mat', doc0) = 1/6 = 0.166667
    matrix = tfidf(DOCS)
    vocab = sorted({tok for doc in DOCS for tok in doc.lower().split()})
    idx = {term: i for i, term in enumerate(vocab)}

    idf_the = math.log(3 / 2) + 1.0
    idf_cat = math.log(3 / 1) + 1.0
    tfidf_the = (2 / 6) * idf_the
    tfidf_cat = (1 / 6) * idf_cat
    tfidf_mat = (1 / 6) * idf_cat  # df(mat) == df(cat) == 1

    assert matrix[0, idx["the"]] == pytest.approx(tfidf_the, rel=1e-9)
    assert matrix[0, idx["cat"]] == pytest.approx(tfidf_cat, rel=1e-9)
    assert matrix[0, idx["mat"]] == pytest.approx(tfidf_mat, rel=1e-9)
    assert matrix[0, idx["the"]] == pytest.approx(0.4684883693693881, rel=1e-6)


def test_tfidf_common_word_scores_lower_than_rare_word_same_tf():
    matrix = tfidf(DOCS)
    vocab = sorted({tok for doc in DOCS for tok in doc.lower().split()})
    idx = {term: i for i, term in enumerate(vocab)}
    # 'sat' (df=2, common) and 'cat' (df=1, rare) each occur once in doc0, so
    # their tf is identical (1/6) — isolating idf, 'cat' must score higher.
    assert matrix[0, idx["cat"]] > matrix[0, idx["sat"]]


def test_tfidf_matrix_shape():
    matrix = tfidf(DOCS)
    vocab_size = len({tok for doc in DOCS for tok in doc.lower().split()})
    assert matrix.shape == (len(DOCS), vocab_size)


def test_tfidf_vectorizer_fit_transform_matches_function():
    vec = TfidfVectorizer()
    fitted = vec.fit_transform(DOCS)
    assert np.allclose(fitted, tfidf(DOCS))
    assert vec.vocab_ == sorted({tok for doc in DOCS for tok in doc.lower().split()})


def test_tfidf_vectorizer_requires_fit_before_transform():
    vec = TfidfVectorizer()
    with pytest.raises(RuntimeError):
        vec.transform(["the cat"])


def test_tfidf_trace_shape_and_fields():
    t = tfidf_trace(DOCS)
    # tokenize, tf, df, idf, weighted matrix
    assert len(t) == 5
    assert np.allclose(t.result, tfidf(DOCS))
    assert "idf" in t.formula
    assert len(t.why_ai) >= 1


def test_tfidf_rejects_empty_docs():
    with pytest.raises(ValueError):
        tfidf_trace([])
    with pytest.raises(ValueError):
        tfidf_trace([""])


# --- Edit distance ------------------------------------------------------------
def test_edit_distance_matches_known_reference_kitten_sitting():
    # Textbook reference value: Levenshtein("kitten", "sitting") == 3
    # (substitute k->s, substitute e->i, insert g).
    assert edit_distance("kitten", "sitting") == 3


def test_edit_distance_identical_strings_is_zero():
    assert edit_distance("cat", "cat") == 0


def test_edit_distance_against_empty_string_is_length():
    assert edit_distance("kitten", "") == 6
    assert edit_distance("", "sitting") == 7


def test_edit_distance_single_insertion():
    assert edit_distance("cat", "cats") == 1


def test_edit_distance_symmetric():
    assert edit_distance("flaw", "lawn") == edit_distance("lawn", "flaw")


def test_edit_script_reconstructs_target():
    # Applying the backtraced ops left-to-right must reproduce b's characters.
    ops = edit_script("cat", "cats")
    rebuilt = "".join(to for op, _frm, to in ops if op != "delete" and to != "-")
    assert rebuilt == "cats"


def test_edit_distance_trace_shape():
    t = edit_distance_trace("kitten", "sitting")
    # base cases, fill table, read distance, backtrace
    assert len(t) == 4
    assert t.result == 3
    assert "min(" in t.formula


def test_edit_distance_rejects_non_strings():
    with pytest.raises(TypeError):
        edit_distance_trace(123, "abc")


# --- N-gram language model -----------------------------------------------------
TRAIN_CORPUS = ["the cat sat on the mat", "the dog sat on the log", "the cat chased the dog"]


def test_ngram_perplexity_matches_hand_computed_reference():
    # Independently hand-computed bigram model with add-1 smoothing over
    # TRAIN_CORPUS (V=10 tokens including <s>/</s>), scored on
    # "the cat sat on the log":
    #   grams: (<s>,the) (the,cat) (cat,sat) (sat,on) (on,the) (the,log) (log,</s>)
    #   probs: 4/13, 3/16, 1/6, 1/4, 1/4, 1/8, 2/11
    probs = [4 / 13, 3 / 16, 1 / 6, 1 / 4, 1 / 4, 1 / 8, 2 / 11]
    expected_ppl = math.exp(-sum(math.log(p) for p in probs) / len(probs))

    model = NGramModel(n=2, k=1.0).fit(TRAIN_CORPUS)
    ppl = model.perplexity(["the cat sat on the log"])
    assert ppl == pytest.approx(expected_ppl, rel=1e-9)
    assert ppl == pytest.approx(4.953859830555323, rel=1e-6)


def test_ngram_prob_sums_to_one_over_vocab():
    model = NGramModel(n=2, k=1.0).fit(TRAIN_CORPUS)
    context = ("the",)
    total = sum(model.prob(context, word) for word in model.vocab)
    assert total == pytest.approx(1.0, rel=1e-9)


def test_ngram_smoothing_gives_unseen_bigram_nonzero_probability():
    model = NGramModel(n=2, k=1.0).fit(TRAIN_CORPUS)
    # "chased" never directly follows "the" in training (0 raw count), but
    # add-k smoothing must still keep its probability strictly above 0.
    p_unseen = model.prob(("the",), "chased")
    assert p_unseen > 0.0


def test_ngram_without_smoothing_zero_count_gives_zero_probability():
    model = NGramModel(n=2, k=0.0).fit(TRAIN_CORPUS)
    assert model.prob(("the",), "chased") == 0.0


def test_ngram_rejects_bad_n():
    with pytest.raises(ValueError):
        NGramModel(n=0)


def test_ngram_rejects_wrong_context_length():
    model = NGramModel(n=3, k=1.0).fit(TRAIN_CORPUS)
    with pytest.raises(ValueError):
        model.prob(("only_one",), "word")


def test_ngram_trace_shape_and_result():
    t = ngram_trace(TRAIN_CORPUS, "the cat sat on the log", n=2, k=1.0)
    # tokenize+pad, count ngrams, conditional probs, perplexity
    assert len(t) == 4
    assert t.result == pytest.approx(4.953859830555323, rel=1e-6)
    assert "PPL" in t.formula
    assert len(t.why_ai) >= 1


def test_ngram_rejects_empty_corpus():
    with pytest.raises(ValueError):
        ngram_trace([], "the cat", n=2)


# --- Byte-pair encoding ---------------------------------------------------------
BPE_CORPUS = ["low", "low", "low", "lower", "lower", "lowest", "newest", "newest", "widest"]


def test_bpe_merges_match_hand_computed_reference():
    # Independently hand-traced (see module docstring's algorithm): with
    # end-of-word marker '</w>', the frequency-ranked merges are:
    #   round1 ('o','w')=6, round2 ('l','ow')=6, round3 ('t','</w>')=4,
    #   round4 ('s','t</w>')=4, round5 ('e','st</w>')=4, round6 ('low','</w>')=3
    tok = BPETokenizer(num_merges=6).train(BPE_CORPUS)
    expected_merges = [
        ("o", "w"),
        ("l", "ow"),
        ("t", "</w>"),
        ("s", "t</w>"),
        ("e", "st</w>"),
        ("low", "</w>"),
    ]
    assert tok.merges == expected_merges


def test_bpe_encode_lowest_matches_hand_computed_reference():
    tok = BPETokenizer(num_merges=6).train(BPE_CORPUS)
    # Replaying the 6 merges above on 'l','o','w','e','s','t','</w>' yields
    # 'low' + 'est</w>' (see hand-trace in module docstring / test above).
    assert tok.encode("lowest") == ["low", "est</w>"]


def test_bpe_encode_word_seen_verbatim_in_training():
    tok = BPETokenizer(num_merges=6).train(BPE_CORPUS)
    assert tok.encode("low") == ["low</w>"]


def test_bpe_more_merges_never_increases_token_count():
    few = BPETokenizer(num_merges=1).train(BPE_CORPUS).encode("lowest")
    many = BPETokenizer(num_merges=6).train(BPE_CORPUS).encode("lowest")
    assert len(many) <= len(few)


def test_bpe_stops_early_when_no_pairs_left():
    # A single-character "corpus" runs out of pairs to merge almost immediately;
    # training must not raise even if num_merges exceeds what's possible.
    tok = BPETokenizer(num_merges=50).train(["a", "a", "a"])
    assert len(tok.merges) < 50


def test_bpe_encode_before_train_raises():
    with pytest.raises(RuntimeError):
        BPETokenizer(num_merges=2).encode("cat")


def test_bpe_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BPETokenizer(num_merges=2).train([])


def test_bpe_trace_shape_and_result():
    t = bpe_trace(BPE_CORPUS, num_merges=6, encode_word="lowest")
    # split, initial vocab, 6 merge rounds, final vocab, encode = 10
    assert len(t) == 10
    assert t.result == ["low", "est</w>"]
    assert len(t.why_ai) >= 1


def test_bpe_rejects_negative_merges():
    with pytest.raises(ValueError):
        bpe_trace(BPE_CORPUS, num_merges=-1, encode_word="low")


# --- Skip-gram word2vec ---------------------------------------------------------
W2V_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat chased the mouse",
    "the dog chased the mouse",
]


def test_skipgram_pairs_match_hand_computed_reference():
    # window=1 over "the cat sat": center 'the' -> context 'cat';
    # center 'cat' -> contexts 'the','sat'; center 'sat' -> context 'cat'.
    pairs = skipgram_pairs(["the cat sat"], window=1)
    assert pairs == [("the", "cat"), ("cat", "the"), ("cat", "sat"), ("sat", "cat")]


def test_skipgram_pairs_rejects_bad_window():
    with pytest.raises(ValueError):
        skipgram_pairs(["the cat sat"], window=0)


def test_word2vec_training_is_deterministic_given_seed():
    t1 = word2vec_trace(W2V_CORPUS, window=1, dim=4, steps=20, lr=0.2, seed=0)
    t2 = word2vec_trace(W2V_CORPUS, window=1, dim=4, steps=20, lr=0.2, seed=0)
    assert t1.meta["losses"] == t2.meta["losses"]


def test_word2vec_softmax_prediction_sums_to_one():
    vocab = ["cat", "dog", "mat", "sat", "the"]
    model = SkipGramModel(vocab, dim=3, seed=0)
    probs = model.predict("cat")
    assert probs.sum() == pytest.approx(1.0, rel=1e-9)
    assert np.all(probs > 0)


def test_word2vec_sgd_step_reduces_loss_on_repeated_pair():
    # Hammering the same (center, context) pair with SGD must monotonically
    # decrease its cross-entropy loss (a basic sanity check of the gradient sign).
    vocab = ["cat", "dog", "mat"]
    model = SkipGramModel(vocab, dim=4, seed=0)
    losses = [model.step("cat", "mat", lr=0.5) for _ in range(10)]
    assert losses[-1] < losses[0]
    assert all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1))


def test_word2vec_step_rejects_unknown_word():
    model = SkipGramModel(["cat", "dog"], dim=2, seed=0)
    with pytest.raises(KeyError):
        model.step("cat", "elephant")


def test_word2vec_most_similar_excludes_query_word():
    model = SkipGramModel(["cat", "dog", "mat"], dim=4, seed=0)
    neighbors = model.most_similar("cat", k=2)
    words = [w for w, _score in neighbors]
    assert "cat" not in words
    assert len(words) == 2


def test_word2vec_rejects_empty_vocab():
    with pytest.raises(ValueError):
        SkipGramModel([], dim=4)


def test_word2vec_trace_shape_and_result():
    t = word2vec_trace(W2V_CORPUS, window=1, dim=4, steps=10, lr=0.2, seed=0)
    # vocab, pairs, init, first-step forward, training loop, neighbors = 6
    assert len(t) == 6
    assert isinstance(t.result, SkipGramModel)
    assert len(t.why_ai) >= 1


def test_word2vec_rejects_corpus_with_no_pairs():
    with pytest.raises(ValueError):
        word2vec_trace(["a"], window=1, steps=5)
