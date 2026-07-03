import numpy as np
import pytest

from optimumai.algebra import Matrix, Vector


def test_dot_result_and_step_count():
    t = Vector([1, 2, 3]).dot_trace(Vector([4, 5, 6]))
    assert t.result == pytest.approx(32.0)
    # one step per component + one summation step
    assert len(t) == 4
    assert t.steps[0].value == pytest.approx(4.0)
    assert t.why_ai  # context is always provided


def test_dot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Vector([1, 2]).dot_trace(Vector([1, 2, 3]))


def test_dot_accepts_plain_iterable():
    assert Vector([1, 0, 0]).dot([0, 1, 0]) == pytest.approx(0.0)


def test_norm():
    t = Vector([3, 4]).norm_trace()
    assert t.result == pytest.approx(5.0)


def test_cosine_similarity_parallel_is_one():
    result = Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]))
    assert result == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    assert Vector([1, 0]).cosine_similarity(Vector([0, 1])) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_raises():
    with pytest.raises(ValueError):
        Vector([0, 0]).cosine_similarity(Vector([1, 1]))


def test_vector_rejects_2d():
    with pytest.raises(ValueError):
        Vector([[1, 2], [3, 4]])


def test_matmul_result_matches_numpy():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    t = Matrix(A).matmul_trace(Matrix(B))
    assert np.allclose(t.result, np.array(A) @ np.array(B))
    # one step per output cell
    assert len(t) == 4


def test_matmul_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Matrix([[1, 2, 3]]).matmul_trace(Matrix([[1, 2]]))


def test_matmul_operator_and_transpose():
    m = Matrix([[1, 2], [3, 4]])
    assert np.allclose(m @ m.T, np.array([[5, 11], [11, 25]]))
