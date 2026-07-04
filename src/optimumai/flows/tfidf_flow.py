"""TF-IDF, stage by stage — from raw counts to a weighted document/term matrix.

``tfidf_flow`` renders the full pipeline behind
:func:`optimumai.nlp.tfidf.tfidf_trace` as an interactive flow:

    corpus -> per-doc term counts -> TF -> document frequency -> IDF -> TF·IDF matrix

All numbers come straight from :mod:`optimumai.nlp.tfidf` (the exact
``tf(t,d) = count(t,d)/|d|`` and smoothed ``idf(t) = log(N/df(t)) + 1``
formulas), so the diagram matches the library's own trace exactly.
"""

from __future__ import annotations

from collections import Counter

from optimumai.flows._shared import (
    HOVER_TOOLTIP_JS,
    arrow,
    flow_controls_html,
    matrix_grid,
    page,
    runtime_script,
    stage_box,
    stage_group_close,
    stage_group_open,
    svg_open,
    write,
)
from optimumai.nlp.tfidf import (
    document_frequency,
    inverse_document_frequency,
    term_frequency,
)

_CSS_EXTRA = """
 #flow-svg{min-width:1250px}
 .flow-stage{transition:opacity .25s}
"""


def _tokenize(doc: str) -> list[str]:
    return doc.lower().split()


def tfidf_flow(
    docs: tuple[str, ...] = (
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends",
    ),
    out: str | None = None,
) -> str:
    """Build the TF -> DF -> IDF -> TF-IDF flow for ``docs`` as self-contained HTML.

    Args:
        docs: A small corpus of documents (short strings). Deliberately mirrors
            :func:`optimumai.nlp.tfidf.demo` — "the" is common but uninformative.
        out: Path to write the HTML to (defaults to ``"tfidf_flow.html"``).

    Returns:
        The path the HTML was written to.
    """
    docs = tuple(docs)
    if not docs:
        raise ValueError("docs must be a non-empty sequence of strings")
    n_docs = len(docs)
    tokenized = [_tokenize(d) for d in docs]
    vocab = sorted({tok for toks in tokenized for tok in toks})
    n_vocab = len(vocab)
    doc_labels = [f"doc{i}" for i in range(n_docs)]

    counts = [Counter(toks) for toks in tokenized]
    tf_rows = [term_frequency(d) for d in docs]
    df = document_frequency(list(docs))
    idf = inverse_document_frequency(list(docs))

    count_matrix = [[counts[i].get(term, 0) for term in vocab] for i in range(n_docs)]
    tf_matrix = [[tf_rows[i].get(term, 0.0) for term in vocab] for i in range(n_docs)]
    idf_row = [idf[term] for term in vocab]
    tfidf_matrix = [[tf_matrix[i][j] * idf_row[j] for j in range(n_vocab)] for i in range(n_docs)]

    stage_defs = [
        ("counts", "1. Term counts"),
        ("tf", "2. TF = count / |d|"),
        ("df", "3. Document frequency"),
        ("idf", "4. IDF = log(N/df) + 1"),
        ("tfidf", "5. TF · IDF matrix"),
    ]
    n_stages = len(stage_defs)
    box_w, box_h, gap = 230, 42, 60
    cell = 30
    svg_w = 40 + n_stages * (box_w + gap)
    svg_h = 90 + (n_docs + 1) * cell + 60
    row_y = 60

    svg_parts = [svg_open(svg_w, svg_h)]
    for idx, (sid, title) in enumerate(stage_defs):
        gx = 20 + idx * (box_w + gap)
        svg_parts.append(stage_group_open(sid, gx, row_y))
        svg_parts.append(stage_box(box_w, box_h, title))
        svg_parts.append('<g transform="translate(0,52)">')

        if sid == "counts":
            svg_parts.append(
                matrix_grid(
                    count_matrix,
                    0,
                    0,
                    cell=cell,
                    row_labels=doc_labels,
                    col_labels=vocab,
                    decimals=0,
                    id_prefix="cnt",
                    tooltip_fn=lambda i, j, v: (
                        f"count({vocab[j]!r}, {doc_labels[i]}) = {int(v)} "
                        f"-- raw occurrences in {doc_labels[i]}: {docs[i]!r}"
                    ),
                )
            )
        elif sid == "tf":
            svg_parts.append(
                matrix_grid(
                    [[round(v, 3) for v in row] for row in tf_matrix],
                    0,
                    0,
                    cell=cell,
                    row_labels=doc_labels,
                    col_labels=vocab,
                    lo=0.0,
                    hi=max(max(row) for row in tf_matrix) or 1.0,
                    id_prefix="tf",
                    tooltip_fn=lambda i, j, v: (
                        f"tf({vocab[j]!r}, {doc_labels[i]}) = {int(count_matrix[i][j])}"
                        f"/{len(tokenized[i])} = {v:.4f} "
                        f"-- fraction of {doc_labels[i]}'s words that are {vocab[j]!r}"
                    ),
                )
            )
        elif sid == "df":
            df_row = [[df.get(term, 0) for term in vocab]]
            svg_parts.append(
                matrix_grid(
                    df_row,
                    0,
                    0,
                    cell=cell,
                    row_labels=["df"],
                    col_labels=vocab,
                    decimals=0,
                    lo=0,
                    hi=n_docs,
                    id_prefix="df",
                    tooltip_fn=lambda i, j, v: (
                        f"df({vocab[j]!r}) = {int(v)} of {n_docs} docs contain it "
                        f"-- higher df means less distinctive"
                    ),
                )
            )
        elif sid == "idf":
            idf_row_disp = [[round(v, 3) for v in idf_row]]
            svg_parts.append(
                matrix_grid(
                    idf_row_disp,
                    0,
                    0,
                    cell=cell,
                    row_labels=["idf"],
                    col_labels=vocab,
                    lo=min(idf_row),
                    hi=max(idf_row),
                    id_prefix="idf",
                    tooltip_fn=lambda i, j, v: (
                        f"idf({vocab[j]!r}) = log({n_docs}/{df.get(vocab[j], 0)}) + 1 "
                        f"= {v:.4f} -- rarer terms across the corpus score higher"
                    ),
                )
            )
        elif sid == "tfidf":
            svg_parts.append(
                matrix_grid(
                    [[round(v, 4) for v in row] for row in tfidf_matrix],
                    0,
                    0,
                    cell=cell,
                    row_labels=doc_labels,
                    col_labels=vocab,
                    id_prefix="tfidf",
                    tooltip_fn=lambda i, j, v: (
                        f"tfidf({vocab[j]!r}, {doc_labels[i]}) = "
                        f"{tf_matrix[i][j]:.4f} * {idf_row[j]:.4f} = {v:.4f} "
                        f"-- frequent here AND rare elsewhere scores highest"
                    ),
                )
            )

        svg_parts.append("</g>")
        svg_parts.append(stage_group_close())
        if idx < n_stages - 1:
            ax = gx + box_w
            ay = row_y + box_h / 2
            svg_parts.append(arrow(ax, ay, ax + gap, ay, idx))

    svg_parts.append("</svg>")
    svg = "\n".join(svg_parts)

    top_term_per_doc = []
    for i, row in enumerate(tfidf_matrix):
        j = max(range(n_vocab), key=lambda jj: row[jj])
        top_term_per_doc.append(f"{doc_labels[i]}: {vocab[j]!r} ({row[j]:.3f})")

    captions = [
        (
            f"<b>Count raw terms.</b> Split each of the {n_docs} documents into words and "
            "count occurrences. A word appearing 3 times in a 6-word document is not "
            "automatically more important than one appearing once in a 3-word document — "
            "that's what the next stage fixes."
        ),
        (
            "<b>Term frequency: tf(t,d) = count(t,d) / |d|.</b> Normalizing by document "
            "length means longer documents don't automatically dominate; tf is the "
            "fraction of a document's words that are this term."
        ),
        (
            f"<b>Document frequency: df(t) = docs containing t.</b> Out of {n_docs} "
            "documents, how many mention this word at all? A word in every document "
            "(like 'the') says nothing about topic."
        ),
        (
            "<b>Smoothed IDF: idf(t) = log(N/df(t)) + 1.</b> Rare, distinctive words get "
            "a high multiplier; ubiquitous words get pulled toward the floor of 1.0 "
            "(the '+1' keeps a word in every document from being zeroed out entirely)."
        ),
        (
            "<b>TF · IDF.</b> Multiply the two opposing forces: local frequency rewards "
            "repetition, global rarity rewards distinctiveness. Highest scorer per "
            f"document: {'; '.join(top_term_per_doc)}."
        ),
    ]
    stages = [
        {"id": sid, "title": title.split(". ", 1)[1], "caption": cap}
        for (sid, title), cap in zip(stage_defs, captions, strict=True)
    ]

    body = f"""
<h1>TF-IDF flow</h1>
<p class="sub">How {n_docs} documents ({n_vocab} unique words) become a
document x term weight matrix. Step through the pipeline, or hover any cell
for the exact value and why it landed there.</p>
{flow_controls_html()}
<div id="flow-wrap">
{svg}
</div>
<p class="legend">tfidf(t,d) = tf(t,d) * idf(t), idf(t) = log(N/df(t)) + 1
(smooth idf, matching scikit-learn's default). Documents:
{" | ".join(f"{doc_labels[i]}={docs[i]!r}" for i in range(n_docs))}</p>
"""
    script = runtime_script(HOVER_TOOLTIP_JS, stages)
    html = page(
        title="OptimumAI — TF-IDF flow",
        heading_sr=(
            "Interactive TF-IDF pipeline: step through raw term counts, term "
            "frequency, document frequency, inverse document frequency, and the "
            "final weighted matrix, hovering any cell for its exact value."
        ),
        body=body,
        script=script,
        css_extra=_CSS_EXTRA,
    )
    return write(html, out, "tfidf_flow.html")
