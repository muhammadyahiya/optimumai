import pytest

from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def test_trace_add_indexes_and_records():
    t = Trace(op="demo")
    t.add("first", "1 + 1 = 2", 2).add("second", "2 + 2 = 4", 4)
    assert len(t) == 2
    assert [s.index for s in t] == [1, 2]
    assert t.last().value == 4


def test_explain_level_parsing_and_order():
    assert ExplainLevel.parse("engineer") is ExplainLevel.ENGINEER
    assert ExplainLevel.parse(ExplainLevel.BEGINNER) is ExplainLevel.BEGINNER
    assert ExplainLevel.RESEARCHER.at_least(ExplainLevel.BEGINNER)
    assert not ExplainLevel.BEGINNER.at_least(ExplainLevel.ENGINEER)


def test_explain_level_rejects_garbage():
    with pytest.raises(ValueError):
        ExplainLevel.parse("wizard")


def test_render_returns_result(capsys):
    t = Trace(op="demo", result=42, formula="x = 42")
    t.add("only step", "42", 42)
    out = t.render("beginner")
    assert out == 42
    captured = capsys.readouterr().out
    assert "DEMO" in captured
