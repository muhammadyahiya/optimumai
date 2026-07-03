"""Classical NLP — the statistical machinery beneath modern language models.

Before attention, before embeddings even, NLP was tokenization, counting, and
dynamic programming. This package covers the fundamentals that still power
production search/retrieval systems and explain *why* the neural approach in
:mod:`optimumai.transformers` and :mod:`optimumai.embeddings` works:

* :mod:`optimumai.nlp.bpe` — byte-pair encoding, how text becomes tokens
* :mod:`optimumai.nlp.tfidf` — TF-IDF, weighting words by how distinguishing they are
* :mod:`optimumai.nlp.ngram` — n-gram language models, counting your way to "next word"
* :mod:`optimumai.nlp.edit_distance` — Levenshtein distance via dynamic programming
* :mod:`optimumai.nlp.word2vec` — skip-gram, the seed idea behind every learned embedding
"""

from optimumai.nlp.bpe import BPETokenizer, bpe_trace
from optimumai.nlp.edit_distance import edit_distance, edit_distance_trace, edit_script
from optimumai.nlp.ngram import NGramModel, ngram_trace, perplexity
from optimumai.nlp.tfidf import TfidfVectorizer, tfidf, tfidf_trace
from optimumai.nlp.word2vec import SkipGramModel, skipgram_pairs, word2vec_trace

__all__ = [
    "BPETokenizer",
    "NGramModel",
    "SkipGramModel",
    "TfidfVectorizer",
    "bpe_trace",
    "edit_distance",
    "edit_distance_trace",
    "edit_script",
    "ngram_trace",
    "perplexity",
    "skipgram_pairs",
    "tfidf",
    "tfidf_trace",
    "word2vec_trace",
]
