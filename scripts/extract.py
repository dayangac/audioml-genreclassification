"""Extract features for the fault-filtered split into data/gtzan/features/{split}.npz."""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402

DATA = ROOT / "data" / "gtzan"
SPLITS = ("train", "valid", "test")


def _extract_one(rel):
    try:
        return rel, F.extract(DATA / "genres" / rel), None
    except Exception as exc:  # decoder errors on corrupt files
        return rel, None, f"{type(exc).__name__}: {exc}"


def main():
    out = DATA / "features"
    out.mkdir(exist_ok=True)
    genres = sorted(p.name for p in (DATA / "genres").iterdir() if p.is_dir())
    skipped = []
    with ProcessPoolExecutor() as pool:
        for split in SPLITS:
            rels = (DATA / "splits" / f"{split}_filtered.txt").read_text().split()
            rows = []
            for rel, feats, err in pool.map(_extract_one, rels, chunksize=4):
                if err:
                    skipped.append(f"{split}\t{rel}\t{err}")
                    continue
                rows.append((rel, feats))
            np.savez_compressed(
                out / f"{split}.npz",
                logmel=np.stack([f["logmel"] for _, f in rows]),
                chroma=np.stack([f["chroma"] for _, f in rows]),
                summary=np.stack([f["summary"] for _, f in rows]),
                labels=np.array([genres.index(r.split("/")[0]) for r, _ in rows]),
                paths=np.array([r for r, _ in rows]),
                names=np.array(rows[0][1]["names"]),
                genres=np.array(genres),
            )
            print(f"{split}: {len(rows)} clips")
    (out / "skipped.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""))
    print(f"skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
