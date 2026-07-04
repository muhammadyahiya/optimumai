"""Tests for the self-contained HTML playgrounds (no browser required).

Each generator is checked for: the file actually gets written, the HTML is
substantial, it contains the expected structural marker (canvas/svg/script),
a concept-specific marker, and evidence that Python-computed numbers were
embedded as JSON for the JS to consume. The dispatcher is checked for
correct routing and for raising on unknown names.
"""

from __future__ import annotations

import json

import pytest

from optimumai.visualization.playgrounds import (
    astar_playground,
    kmeans_playground,
    playground,
    transformer_attention_playground,
)


def test_attention_playground_writes_file_with_expected_content(tmp_path):
    out = str(tmp_path / "attn.html")
    path = transformer_attention_playground("the cat sat on the mat", out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 800
    assert "<script" in html
    assert "attention" in html.lower()
    assert "temperature" in html.lower()

    tokens = "the cat sat on the mat".split()
    for tok in tokens:
        assert tok in html


def test_attention_playground_embeds_score_matrix_as_json(tmp_path):
    out = str(tmp_path / "attn.html")
    transformer_attention_playground("a b c", out=out)
    html = _read(out)

    tokens_json = _extract_json(html, "TOKENS =")
    assert tokens_json == ["a", "b", "c"]

    scores_json = _extract_json(html, "SCORES =")
    assert isinstance(scores_json, list)
    assert len(scores_json) == 3
    assert all(len(row) == 3 for row in scores_json)


def test_attention_playground_default_out_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = transformer_attention_playground()
    assert path == "transformer_attention_playground.html"
    assert (tmp_path / path).exists()


def test_attention_playground_handles_empty_text(tmp_path):
    out = str(tmp_path / "empty.html")
    path = transformer_attention_playground("   ", out=out)
    html = _read(path)
    assert len(html) > 800
    assert "<empty>" in html


def test_kmeans_playground_writes_file_with_expected_content(tmp_path):
    out = str(tmp_path / "kmeans.html")
    path = kmeans_playground(out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 800
    assert "<canvas" in html
    assert "kmeans" in html.lower() or "k-means" in html.lower()
    assert "centroid" in html.lower()
    assert "inertia" in html.lower()

    seed_points = _extract_kmeans_points(html)
    assert len(seed_points) > 0
    assert all(len(p) == 2 for p in seed_points)


def test_kmeans_playground_has_controls(tmp_path):
    out = str(tmp_path / "kmeans.html")
    kmeans_playground(out=out)
    html = _read(out)
    for marker in ("Add random", "Step", "Run", "Reset", "kSelect"):
        assert marker in html


def test_astar_playground_writes_file_with_expected_content(tmp_path):
    out = str(tmp_path / "astar.html")
    path = astar_playground(out=out)
    assert path == out

    html = _read(out)
    assert len(html) > 800
    assert "<canvas" in html
    assert "astar" in html.lower() or "a*" in html.lower()
    assert "heuristic" in html.lower()
    assert "Manhattan" in html

    assert "const COLS = " in html
    assert "ROWS = " in html


def test_astar_playground_has_controls(tmp_path):
    out = str(tmp_path / "astar.html")
    astar_playground(out=out)
    html = _read(out)
    for marker in ("Run", "Reset", "modeSelect", "expandedLabel", "pathLabel"):
        assert marker in html


def test_playground_dispatcher_routes_correctly(tmp_path):
    attn_out = str(tmp_path / "a.html")
    kmeans_out = str(tmp_path / "k.html")
    astar_out = str(tmp_path / "s.html")

    assert playground("attention", out=attn_out) == attn_out
    assert playground("kmeans", out=kmeans_out) == kmeans_out
    assert playground("astar", out=astar_out) == astar_out

    assert "attention" in _read(attn_out).lower()
    assert "centroid" in _read(kmeans_out).lower()
    assert "Manhattan" in _read(astar_out)


def test_playground_unknown_name_raises():
    with pytest.raises(ValueError, match="attention"):
        playground("nope")


def test_playground_unknown_name_lists_valid_choices():
    with pytest.raises(ValueError, match="kmeans"):
        playground("bogus")


# --- helpers ----------------------------------------------------------------


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _extract_json(html: str, marker: str):
    """Pull the JSON literal that immediately follows ``marker`` in the JS."""
    idx = html.index(marker) + len(marker)
    rest = html[idx:].lstrip()
    end = rest.index(";")
    return json.loads(rest[:end].rstrip().rstrip(";"))


def _extract_kmeans_points(html: str):
    marker = "let points = "
    idx = html.index(marker) + len(marker)
    rest = html[idx:]
    end = rest.index(".map(")
    return json.loads(rest[:end])
