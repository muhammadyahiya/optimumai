import numpy as np
import pytest

from optimumai.world_models import JEPA


def test_jepa_energy_is_nonnegative_scalar():
    energy = JEPA.demo().result
    assert isinstance(energy, float)
    assert energy >= 0.0


def test_jepa_demo_is_reproducible():
    assert JEPA.demo(seed=3).result == pytest.approx(JEPA.demo(seed=3).result)


def test_jepa_identical_views_have_lower_energy_than_random():
    jepa = JEPA(input_dim=6, embed_dim=4, seed=0)
    v = np.ones(6)
    same = jepa.forward(v, v)
    different = jepa.forward(v, -v)
    assert same <= different  # predicting an identical view is no harder


def test_jepa_rejects_wrong_input_dim():
    with pytest.raises(ValueError):
        JEPA(input_dim=6, embed_dim=4).trace(np.ones(5), np.ones(6))


def test_jepa_trace_has_four_stages():
    assert len(JEPA.demo()) == 4  # encode x, encode y, predict, energy
