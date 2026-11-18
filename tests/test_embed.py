import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from embed import CLIP_S, HOP_S, WIN_S, windows  # noqa: E402

SR = 10  # small rate keeps the arrays tiny


def test_full_clip_gives_five_overlapping_windows():
    y = np.arange(CLIP_S * SR, dtype=float)
    w = windows(y, SR)
    assert len(w) == 1 + (CLIP_S - WIN_S) // HOP_S == 5
    assert all(len(x) == WIN_S * SR for x in w)
    assert [x[0] for x in w] == [i * HOP_S * SR for i in range(5)]


def test_short_clip_is_zero_padded():
    w = windows(np.ones(25 * SR), SR)
    assert len(w) == 5
    assert w[-1][-1] == 0.0 and w[0][0] == 1.0


def test_long_clip_is_cut():
    y = np.arange(40 * SR, dtype=float)
    w = windows(y, SR)
    assert len(w) == 5
    assert w[-1][-1] == CLIP_S * SR - 1
