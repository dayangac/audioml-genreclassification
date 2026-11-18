import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_dataset import SEG_S, SR, segments  # noqa: E402


def test_segments_are_under_five_seconds():
    assert SEG_S < 5
    assert all(len(s) / SR < 5 for s in segments(np.zeros(30 * SR), SR))


def test_thirty_second_clip_gives_ten_three_second_segments():
    segs = segments(np.zeros(30 * SR), SR, seconds=3.0)
    assert len(segs) == 10
    assert all(len(s) == 3 * SR for s in segs)


def test_segments_do_not_overlap_and_keep_order():
    y = np.arange(30 * SR, dtype=float)
    segs = segments(y, SR, seconds=3.0)
    np.testing.assert_array_equal(np.concatenate(segs), y)


def test_short_leftover_is_dropped():
    segs = segments(np.zeros(int(29.93 * SR)), SR, seconds=3.0)
    assert len(segs) == 9


@pytest.mark.parametrize("n", [0, SR])
def test_clip_shorter_than_one_segment_gives_none(n):
    assert segments(np.zeros(n), SR, seconds=3.0) == []
