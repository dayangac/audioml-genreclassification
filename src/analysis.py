"""Helpers for analysing the clip-level summary features."""

import numpy as np
from sklearn.metrics import accuracy_score


def feature_family(name):
    """'mfcc_mfcc3_mean' -> 'mfcc', 'tempo_bpm' -> 'tempo'."""
    return name.split("_")[0]


def group_permutation_importance(model, X, y, groups, n_repeats=20, seed=0):
    """Accuracy drop (mean, std) per group when its columns are shuffled together."""
    rng = np.random.default_rng(seed)
    base = accuracy_score(y, model.predict(X))
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - accuracy_score(y, model.predict(Xp)))
        out[name] = (float(np.mean(drops)), float(np.std(drops)))
    return out
