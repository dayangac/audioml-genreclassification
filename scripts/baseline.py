"""SVM baseline and feature-group ablation; writes outputs/baseline.md."""

from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
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


def load(split):
    d = np.load(FEAT / f"{split}.npz")
    return d["summary"], d["labels"], list(d["names"]), list(d["genres"])


def columns(names, groups):
    prefixes = tuple(p + "_" for g in groups for p in GROUPS[g])
    return [i for i, n in enumerate(names) if n.startswith(prefixes)]


def fit_select(Xtr, ytr, Xva, yva):
    best = None
    for C, gamma in product([0.1, 1, 10, 100], ["scale", 0.001, 0.01]):
        model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)).fit(Xtr, ytr)
        acc = accuracy_score(yva, model.predict(Xva))
        if best is None or acc > best[0]:
            best = (acc, model, C, gamma)
    return best


def main():
    Xtr, ytr, names, genres = load("train")
    Xva, yva, _, _ = load("valid")
    Xte, yte, _, _ = load("test")

    lines = ["# SVM baseline, GTZAN fault-filtered split", "",
             f"train {len(ytr)} / valid {len(yva)} / test {len(yte)} clips, "
             f"{len(names)} summary features", "",
             "| features | dims | C | gamma | valid acc | test acc |",
             "|---|---|---|---|---|---|"]
    full_model = None
    for label, groups in [("all", tuple(GROUPS)), ("timbre", ("timbre",)),
                          ("harmony", ("harmony",)), ("rhythm", ("rhythm",)),
                          ("timbre + harmony", ("timbre", "harmony"))]:
        cols = columns(names, groups)
        va, model, C, gamma = fit_select(Xtr[:, cols], ytr, Xva[:, cols], yva)
        te = accuracy_score(yte, model.predict(Xte[:, cols]))
        lines.append(f"| {label} | {len(cols)} | {C} | {gamma} | {va:.3f} | {te:.3f} |")
        print(lines[-1])
        if label == "all":
            full_model, full_cols = model, cols

    cm = confusion_matrix(yte, full_model.predict(Xte[:, full_cols]))
    lines += ["", "## Test confusion matrix (all features; rows = true)", "",
              "| | " + " | ".join(g[:4] for g in genres) + " |",
              "|---" * (len(genres) + 1) + "|"]
    lines += [f"| {g} | " + " | ".join(str(v) for v in row) + " |"
              for g, row in zip(genres, cm)]
    per_class = cm.diagonal() / cm.sum(axis=1)
    lines += ["", "Per-genre test recall: " +
              ", ".join(f"{g} {r:.2f}" for g, r in zip(genres, per_class))]
    print(lines[-1])

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "baseline.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
