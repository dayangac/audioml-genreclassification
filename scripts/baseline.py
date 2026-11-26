"""SVM baseline and feature-group ablation; writes outputs/baseline.md."""
# this fulfills the requirement of a baseline model from the document

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "gtzan" / "features"

GROUPS = {
    "timbre": ("mfcc", "dmfcc", "centroid", "bandwidth", "rolloff", "flatness",
               "contrast", "zcr", "rms"),
    "harmony": ("chroma", "tonnetz"),
    "rhythm": ("onset", "tempo"),
}

FEATURE_SETS = {
    "all": tuple(GROUPS),
    "timbre": ("timbre",),
    "harmony": ("harmony",),
    "rhythm": ("rhythm",),
    "timbre + harmony": ("timbre", "harmony"),
}


def load(split):
    """load one of the splits train, test, val"""
    d = np.load(FEAT / f"{split}.npz")
    return d["summary"], d["labels"], list(d["names"]), list(d["genres"])


def columns(names, groups):
    """get which features belong to chosen group""" 
    prefixes = tuple(p + "_" for g in groups for p in GROUPS[g])
    return [i for i, n in enumerate(names) if n.startswith(prefixes)]


def fit_select(Xtr, ytr, Xva, yva):
    """try multiple setting of the support vector machine, keep the best"""
    best = None
    # try different C's and gamms
    # C for how much  muistakes are punished
    # gamma for how much similar clips influence results
    for C, gamma in product([0.1, 1, 10, 100], ["scale", 0.001, 0.01]):
        model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)).fit(Xtr, ytr)
        acc = accuracy_score(yva, model.predict(Xva))
        if best is None or acc > best[0]:
            best = (acc, model, C, gamma)
    return best


def run_feature_sets(names, Xtr, ytr, Xva, yva, Xte, yte):
    """train and score one SVM per feature set"""
    rows, preds = [], {}
    for label, groups in FEATURE_SETS.items():
        # get columns in that set
        cols = columns(names, groups)
        # chose the settings via fit_select
        va, model, C, gamma = fit_select(Xtr[:, cols], ytr, Xva[:, cols], yva)
        # keep labels so confusion matrix can use them
        preds[label] = model.predict(Xte[:, cols])
        # accuracy on test
        te = accuracy_score(yte, preds[label])
        # results appeneded to therows array
        rows.append(f"| {label} | {len(cols)} | {C} | {gamma} | {va:.3f} | {te:.3f} |")
        print(rows[-1])
    return rows, preds


def header(n_train, n_valid, n_test, n_features):
    """title and top of the results table"""
    # in short a bit of TUI'ing
    return ["# SVM baseline, GTZAN fault-filtered split", "",
            f"train {n_train} / valid {n_valid} / test {n_test} clips, "
            f"{n_features} summary features", "",
            "| features | dims | C | gamma | valid acc | test acc |",
            "|---|---|---|---|---|---|"]


def confusion_report(y_true, y_pred, genres):
    """confusion matrix and per-genre recall"""
    # easy confusion matrix via help of pandas and some TUI'ing
    cm = confusion_matrix(y_true, y_pred)
    table = pd.DataFrame(cm, index=genres, columns=[g[:4] for g in genres])
    recall = recall_score(y_true, y_pred, average=None)
    return ["", "## Test confusion matrix (all features; rows = true)", "",
            "```", table.to_string(), "```",
            "", "Per-genre test recall: " +
            ", ".join(f"{g} {r:.2f}" for g, r in zip(genres, recall))]


def main():
    """main function that combines all to load data run experiments and save reports """
    Xtr, ytr, names, genres = load("train")
    Xva, yva, _, _ = load("valid")
    Xte, yte, _, _ = load("test")

    # single Support Vector Machine per feature set
    rows, preds = run_feature_sets(names, Xtr, ytr, Xva, yva, Xte, yte)
    report = confusion_report(yte, preds["all"], genres)
    print(report[-1])
    # join the three parts of the report
    lines = header(len(ytr), len(yva), len(yte), len(names)) + rows + report
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "baseline.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
