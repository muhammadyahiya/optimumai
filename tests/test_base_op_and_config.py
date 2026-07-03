import numpy as np

from optimumai.config import Settings, get_settings
from optimumai.core.explain import ExplainLevel
from optimumai.transformers import Attention


def _qkv():
    rng = np.random.default_rng(2)
    return (rng.normal(size=(2, 4)) for _ in range(3))


def test_base_op_call_returns_result_without_rendering(capsys):
    Q, K, V = _qkv()
    out = Attention()(Q, K, V)  # __call__ without explain
    assert out.shape == (2, 4)
    assert capsys.readouterr().out == ""  # nothing printed


def test_base_op_call_explain_prints(capsys):
    Q, K, V = _qkv()
    out = Attention()(Q, K, V, explain=True, level="beginner")
    assert out.shape == (2, 4)
    assert "ATTENTION" in capsys.readouterr().out


def test_base_op_run_and_explain_methods(capsys):
    Q, K, V = _qkv()
    op = Attention()
    assert np.allclose(op.run(Q, K, V), op.trace(Q, K, V).result)
    op.explain(Q, K, V, level=ExplainLevel.ENGINEER)
    assert "ATTENTION" in capsys.readouterr().out


def test_settings_defaults_and_cache():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.explain_level is ExplainLevel.INTERMEDIATE
    assert settings.color is True
    assert get_settings() is settings  # lru_cache returns the same instance
