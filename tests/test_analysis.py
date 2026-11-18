import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import feature_family, group_permutation_importance  # noqa: E402


@pytest.fixture(scope="module")
def informative_and_noise():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    signal = y[:, None] * 3.0 + rng.normal(size=(400, 2))
    noise = rng.normal(size=(400, 3))
    X = np.hstack([signal, noise])
    model = LogisticRegression().fit(X, y)
    groups = {"signal": [0, 1], "noise": [2, 3, 4]}
    return model, X, y, groups


def test_informative_group_has_large_drop(informative_and_noise):
    imp = group_permutation_importance(*informative_and_noise)
    assert imp["signal"][0] > 0.3


def test_noise_group_has_near_zero_drop(informative_and_noise):
    imp = group_permutation_importance(*informative_and_noise)
    assert abs(imp["noise"][0]) < 0.02


def test_input_is_not_modified(informative_and_noise):
    model, X, y, groups = informative_and_noise
    before = X.copy()
    group_permutation_importance(model, X, y, groups, n_repeats=2)
    np.testing.assert_array_equal(X, before)


def test_seed_makes_result_reproducible(informative_and_noise):
    a = group_permutation_importance(*informative_and_noise, n_repeats=3, seed=7)
    b = group_permutation_importance(*informative_and_noise, n_repeats=3, seed=7)
    assert a == b


@pytest.mark.parametrize("name, family", [
    ("mfcc_mfcc3_mean", "mfcc"), ("dmfcc_dmfcc0_std", "dmfcc"),
    ("tempo_bpm", "tempo"), ("centroid_mean", "centroid"),
])
def test_feature_family(name, family):
    assert feature_family(name) == family
