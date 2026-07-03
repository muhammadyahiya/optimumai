"""Byte-Pair Encoding — how LLMs turn text into a vocabulary of "sub-words."

**Intuition.** Two extremes for tokenizing text both fail: a *character*
vocabulary is tiny but every sequence becomes enormous (slow, and each token
carries almost no meaning), while a *whole-word* vocabulary makes sequences
short but explodes in size and can't handle a word it has never seen ("un" +
"believable" + "ly" all become one unknown blob). BPE finds a middle ground
automatically: start from characters, then repeatedly glue together whichever
adjacent pair of symbols is most common in the training data, growing a
vocabulary of frequent chunks — often whole common words, but sub-word pieces
for rarer ones.

**Math / algorithm.** Given a corpus, first split every word into characters
plus an explicit end-of-word marker (so the model can tell "est" in "est-imate"
apart from the "est" ending "greatest"):

    1. vocab = every distinct symbol seen so far (initially: characters)
    2. repeat `num_merges` times:
       a. count every adjacent symbol pair across all words, weighted by word frequency
       b. pick the single most frequent pair (a, b)
       c. merge: replace every occurrence of the sequence "a b" with the new symbol "ab"
       d. add "ab" to the vocabulary; record the merge rule (a, b) -> ab
    3. encoding new text = apply the learned merge rules in the order they were learned

Each merge is a greedy, deterministic choice — there is no probability model,
just "count pairs, merge the winner, repeat." This is why BPE training is fast
and BPE encoding of new text is just a lookup-and-replay of the recorded merge
list (in learned order) until no more rules apply.

**Why modern LLMs still rest on this.** Essentially every modern LLM
tokenizer (GPT's, LLaMA's, etc.) is a byte-level variant of exactly this
algorithm — it gives them a fixed, closed vocabulary (no true "unknown word"
token, since you can always fall back further toward raw bytes/characters)
while keeping common words as single, information-dense tokens. It is the
first thing that happens to your prompt, before embedding, before attention,
before anything else — see :mod:`optimumai.transformers.text_pipeline` for
the whitespace-tokenized version of that full pipeline.
"""

from __future__ import annotations

from collections import Counter

from optimumai.core.trace import Trace

_EOW = "</w>"  # end-of-word marker so "est" mid-word != "est" at a word boundary


def _word_to_symbols(word: str) -> tuple[str, ...]:
    """Split a word into characters plus a trailing end-of-word marker."""
    return (*word, _EOW)


def _pair_counts(corpus: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """Count every adjacent symbol pair across the corpus, weighted by word frequency."""
    counts: Counter[tuple[str, str]] = Counter()
    for symbols, freq in corpus.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _merge_pair(
    corpus: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int]:
    """Replace every occurrence of ``pair`` with its concatenation across the corpus."""
    merged = "".join(pair)
    new_corpus: dict[tuple[str, ...], int] = {}
    for symbols, freq in corpus.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_corpus[tuple(new_symbols)] = freq
    return new_corpus


class BPETokenizer:
    """Learn BPE merges from a corpus, then encode new words with them.

    Args:
        num_merges: How many merge rules to learn during :meth:`train`.
    """

    def __init__(self, num_merges: int = 10) -> None:
        if num_merges < 0:
            raise ValueError(f"num_merges must be >= 0, got {num_merges}")
        self.num_merges = num_merges
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()

    def train(self, corpus: list[str]) -> BPETokenizer:
        """Learn up to ``num_merges`` merge rules from a list of training words."""
        if not corpus:
            raise ValueError("corpus must be a non-empty list of words")
        word_freq = Counter(corpus)
        symbol_corpus = {_word_to_symbols(w): f for w, f in word_freq.items()}
        self.vocab = {sym for symbols in symbol_corpus for sym in symbols}

        for _ in range(self.num_merges):
            pairs = _pair_counts(symbol_corpus)
            if not pairs:
                break
            best_pair = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
            symbol_corpus = _merge_pair(symbol_corpus, best_pair)
            self.merges.append(best_pair)
            self.vocab.add("".join(best_pair))
        return self

    def encode(self, word: str) -> list[str]:
        """Apply learned merges (in training order) to tokenize ``word``."""
        if not self.merges and not self.vocab:
            raise RuntimeError("call train(...) before encode(...)")
        symbols = list(_word_to_symbols(word))
        for pair in self.merges:
            merged = "".join(pair)
            i = 0
            new_symbols: list[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols


def bpe_trace(corpus: list[str], num_merges: int, encode_word: str) -> Trace:
    """Train BPE on ``corpus`` and trace each merge round, then encode ``encode_word``."""
    if not corpus:
        raise ValueError("corpus must be a non-empty list of words")
    if num_merges < 0:
        raise ValueError(f"num_merges must be >= 0, got {num_merges}")
    if not encode_word:
        raise ValueError("encode_word must be a non-empty string")

    word_freq = Counter(corpus)
    symbol_corpus = {_word_to_symbols(w): f for w, f in word_freq.items()}
    initial_vocab = sorted({sym for symbols in symbol_corpus for sym in symbols})

    t = Trace(
        op="bpe",
        formula="repeat: merge the most frequent adjacent symbol pair into one new symbol",
        complexity=f"O(num_merges * corpus size) = O({num_merges} * {len(corpus)}) for training",
        why_ai=[
            "Every modern LLM tokenizer (GPT, LLaMA, ...) is a byte-level variant of this "
            "exact merge-the-most-frequent-pair algorithm",
            "Gives a fixed, closed vocabulary with no true out-of-vocabulary token — rare "
            "words fall back to smaller sub-word or character pieces",
            "Common words stay single, information-dense tokens while rare ones get spelled "
            "out — a learned compromise between character- and word-level tokenization",
        ],
        meta={"num_merges": num_merges, "corpus_size": len(corpus)},
    )

    split_words = ((w, _word_to_symbols(w)) for w in word_freq)
    t.add(
        f"Split words into characters + end-of-word marker ({_EOW!r})",
        ", ".join(f"{w}={list(symbols)}" for w, symbols in split_words),
        symbol_corpus,
    )
    t.add(
        f"Initial character vocabulary ({len(initial_vocab)} symbols)",
        str(initial_vocab),
        initial_vocab,
    )

    merges: list[tuple[str, str]] = []
    for round_idx in range(num_merges):
        pairs = _pair_counts(symbol_corpus)
        if not pairs:
            t.add(
                f"Round {round_idx + 1}: no adjacent pairs left",
                "stopping early — every word has been merged into a single symbol",
                detail="Training stops once there is nothing left to merge, even if "
                "num_merges hasn't been reached.",
            )
            break
        best_pair, best_count = max(pairs.items(), key=lambda item: (item[1], item[0]))
        symbol_corpus = _merge_pair(symbol_corpus, best_pair)
        merges.append(best_pair)
        top_pairs = ", ".join(f"{p}={c}" for p, c in pairs.most_common(3))
        t.add(
            f"Round {round_idx + 1}: most frequent pair",
            f"pair counts (top 3): {top_pairs}  →  merge {best_pair} "
            f"(count={best_count}) into {''.join(best_pair)!r}",
            best_pair,
            detail=f"Corpus after merge: {[list(s) for s in symbol_corpus]}",
        )

    final_vocab = sorted({sym for symbols in symbol_corpus for sym in symbols})
    t.add(
        f"Final vocabulary ({len(final_vocab)} symbols after {len(merges)} merges)",
        str(final_vocab),
        final_vocab,
    )

    tokenizer = BPETokenizer(num_merges=num_merges)
    tokenizer.merges = merges
    tokenizer.vocab = set(final_vocab)
    encoded = tokenizer.encode(encode_word)
    t.add(
        f"Encode {encode_word!r} by replaying the learned merges in order",
        f"{list(_word_to_symbols(encode_word))}  →  {encoded}",
        encoded,
        detail="New text is tokenized by applying every learned merge rule, in the order "
        "it was learned, to the character-split word.",
    )

    t.result = encoded
    t.meta["merges"] = merges
    t.meta["vocab"] = final_vocab
    return t


def demo(seed: int = 0) -> Trace:
    """Learn 6 merges from a tiny corpus where 'low'/'lower'/'lowest' share a stem."""
    corpus = ["low", "low", "low", "lower", "lower", "lowest", "newest", "newest", "widest"]
    return bpe_trace(corpus, num_merges=6, encode_word="lowest")
