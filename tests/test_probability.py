import numpy as np
import pytest

from optimumai.probability import softmax, softmax_trace


def test_softmax_sums_to_one():
    probs = softmax([2.0, 1.0, 0.1])
    assert np.sum(probs) == pytest.approx(1.0)
    assert np.all(probs > 0)


def test_softmax_is_monotonic_with_input():
    probs = softmax([3.0, 1.0, 0.0])
    # largest logit -> largest probability
    assert np.argmax(probs) == 0


def test_softmax_matches_reference_implementation():
    x = np.array([1.0, 2.0, 3.0])
    ref = np.exp(x) / np.exp(x).sum()
    assert np.allclose(softmax(x), ref)


def test_temperature_below_one_sharpens():
    base = softmax([2.0, 1.0, 0.1], temperature=1.0)
    sharp = softmax([2.0, 1.0, 0.1], temperature=0.5)
    assert sharp.max() > base.max()


def test_softmax_trace_step_shape():
    t = softmax_trace([2.0, 1.0, 0.1])
    # subtract-max, exponentiate, sum, normalize
    assert len(t) == 4
    assert np.sum(t.result) == pytest.approx(1.0)


def test_softmax_rejects_bad_temperature():
    with pytest.raises(ValueError):
        softmax_trace([1.0, 2.0], temperature=0.0)
