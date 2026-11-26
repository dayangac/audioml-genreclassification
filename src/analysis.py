"""Helpers for analysing the clip-level summary features."""

import numpy as np
from sklearn.metrics import accuracy_score


def feature_family(name):
    """takes a feature and returns first part of name, its family"""
    return name.split("_")[0]


def group_permutation_importance(model, X, y, groups, n_repeats=20, seed=0):
    """How much model worsens if one group of features is shuffled"""
    ## setup
    rng = np.random.default_rng(seed) # random num gen
    base = accuracy_score(y, model.predict(X)) # accuracy pre-shuffle
    out = {} # results
    for name, cols in groups.items():
        drops = []
        # shuffle and test multiple times
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - accuracy_score(y, model.predict(Xp)))
        out[name] = (float(np.mean(drops)), float(np.std(drops)))
    # return average drops
    return out
