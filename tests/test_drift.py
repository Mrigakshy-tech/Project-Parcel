import numpy as np
import pytest
from src.parcel_utils import population_stability_index

def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=2000)
    current = rng.normal(size=2000)
    assert population_stability_index(reference, current) >= 0

def test_psi_increases_for_shifted_distribution():
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(3, 1, 2000)
    assert population_stability_index(reference, current) > 0.2

def test_psi_rejects_empty_finite_data():
    with pytest.raises(ValueError):
        population_stability_index([np.nan], [1, 2])
