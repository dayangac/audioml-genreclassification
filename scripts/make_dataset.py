"""Cut the fault-filtered split into 3 s, 16 kHz segments under data/gtzan16k/."""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "gtzan"
OUT = ROOT / "data" / "gtzan16k"
SR = 16000
SEG_S = 3.0
SPLITS = ("train", "valid", "test")


def segments(y, sr, seconds=SEG_S):
    """Non-overlapping segments of exactly `seconds`; a shorter leftover is dropped."""
    n = int(round(seconds * sr))
    return [y[i * n:(i + 1) * n] for i in range(len(y) // n)]


def _process(job):
    split, rel = job
    genre, name = rel.split("/")
    stem = name.rsplit(".", 1)[0]
    try:
        y, _ = librosa.load(SRC / "genres" / rel, sr=SR, mono=True)
    except Exception as exc:  # decoder errors on corrupt files
        return split, rel, None, f"{type(exc).__name__}: {exc}"
    out_dir = OUT / "audio" / split / genre
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seg in enumerate(segments(y, SR)):
        path = out_dir / f"{stem}_{i}.wav"
        sf.write(path, np.clip(seg, -1.0, 1.0), SR, subtype="PCM_16")
        rows.append((str(path.relative_to(OUT)), genre, stem, i))
    return split, rel, rows, None


def main():
    jobs = [(split, rel) for split in SPLITS
            for rel in (SRC / "splits" / f"{split}_filtered.txt").read_text().split()]
    rows = {s: [] for s in SPLITS}
    skipped = []
    with ProcessPoolExecutor() as pool:
        for split, rel, r, err in pool.map(_process, jobs, chunksize=4):
            if err:
                skipped.append(f"{split}\t{rel}\t{err}")
            else:
                rows[split] += r
    for split in SPLITS:
        with open(OUT / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "genre", "clip", "segment"])
            w.writerows(rows[split])
        clips = len({r[2] for r in rows[split]})
        print(f"{split}: {clips} clips -> {len(rows[split])} segments")
    (OUT / "skipped.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""))
    print(f"skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
