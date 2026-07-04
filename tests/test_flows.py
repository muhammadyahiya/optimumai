"""Tests for the self-contained interactive flow visualizations (no browser needed).

Each generator is checked for: the file actually gets written, the HTML is
substantial, it contains an inline SVG and a step-control marker, and the
concept's own real numbers are embedded as JSON for the JS to consume. The
dispatcher is checked for correct routing and for raising on unknown names.
"""

from __future__ import annotations

import pytest

from optimumai.flows import attention_flow, flow, tfidf_flow, transformer_flow, word2vec_flow


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _assert_well_formed_flow_page(html: str) -> None:
    """Loose structural sanity checks for a self-contained flow page."""
    assert html.strip().lower().startswith("<!doctype html>")
    assert html.count("<html") == 1
    assert "</html>" in html
    assert "<head>" in html and "</head>" in html
    assert "<body>" in html and "</body>" in html
    assert "<script>" in html and "</script>" in html
    # No CDN / external network dependency: no <script src=...>, <link>, or
    # fetch of a remote resource. (The SVG xmlns URI is a namespace name, not
    # a network fetch, so it is deliberately not checked here.)
    assert "<script src" not in html.lower()
    assert "<link " not in html.lower()
    assert "cdn." not in html.lower()
    assert "fetch(" not in html
    assert "//fonts.googleapis" not in html


def test_transformer_flow_writes_substantial_self_contained_html(tmp_path):
    out = str(tmp_path / "transformer.html")
    path = transformer_flow("the cat sat on the mat", out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 1500
    assert "<svg" in html
    assert "nextBtn" in html  # Step ▶ control marker
    assert "STAGES" in html  # embedded stage/caption JSON
    assert "attention" in html.lower()
    for tok in ("the", "cat", "sat", "on", "mat"):
        assert tok in html
    _assert_well_formed_flow_page(html)


def test_attention_flow_writes_substantial_self_contained_html(tmp_path):
    out = str(tmp_path / "attention.html")
    path = attention_flow(("a", "b", "c"), out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 1500
    assert "<svg" in html
    assert "nextBtn" in html
    assert "STAGES" in html
    assert "softmax" in html.lower()
    for tok in ("a", "b", "c"):
        assert f">{tok}<" in html
    _assert_well_formed_flow_page(html)


def test_tfidf_flow_writes_substantial_self_contained_html(tmp_path):
    out = str(tmp_path / "tfidf.html")
    path = tfidf_flow(out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 1500
    assert "<svg" in html
    assert "nextBtn" in html
    assert "STAGES" in html
    assert "idf" in html.lower()
    assert "cat" in html.lower()
    _assert_well_formed_flow_page(html)


def test_word2vec_flow_writes_substantial_self_contained_html(tmp_path):
    out = str(tmp_path / "word2vec.html")
    path = word2vec_flow(out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 1500
    assert "<svg" in html
    assert "nextBtn" in html
    assert "STAGES" in html
    assert "softmax" in html.lower()
    assert "cat" in html.lower()
    _assert_well_formed_flow_page(html)


def test_dispatcher_routes_to_the_right_builder(tmp_path):
    expected_out = {
        "transformer": str(tmp_path / "t.html"),
        "attention": str(tmp_path / "a.html"),
        "tfidf": str(tmp_path / "f.html"),
        "word2vec": str(tmp_path / "w.html"),
    }
    for name, out in expected_out.items():
        path = flow(name, out=out)
        assert path == out
        html = _read(path)
        assert len(html) > 1500
        assert "<svg" in html


def test_dispatcher_raises_valueerror_on_unknown_name():
    with pytest.raises(ValueError, match="unknown flow"):
        flow("nope")


def test_dispatcher_error_lists_valid_names():
    with pytest.raises(ValueError) as exc_info:
        flow("bogus")
    message = str(exc_info.value)
    for name in ("transformer", "attention", "tfidf", "word2vec"):
        assert name in message


def test_default_output_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert transformer_flow() == "transformer_flow.html"
    assert attention_flow() == "attention_flow.html"
    assert tfidf_flow() == "tfidf_flow.html"
    assert word2vec_flow() == "word2vec_flow.html"
    for name in (
        "transformer_flow.html",
        "attention_flow.html",
        "tfidf_flow.html",
        "word2vec_flow.html",
    ):
        assert (tmp_path / name).exists()
