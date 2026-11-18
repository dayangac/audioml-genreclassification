import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402


def tone(freq, seconds=F.DURATION_S, amp=0.5):
    t = np.arange(int(F.SR * seconds)) / F.SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def clicks(bpm, seconds=F.DURATION_S):
    y = np.zeros(int(F.SR * seconds), dtype=np.float32)
    step = int(round(F.SR * 60 / bpm))
    for start in range(0, len(y) - 200, step):
        y[start:start + 200] = np.hanning(200)
    return y


@pytest.fixture(scope="module")
def a440():
    y = tone(440.0)
    frames = F.frame_features(y)
    vec, names = F.summary_vector(y, frames)
    return y, frames, dict(zip(names, vec))


def test_fix_length_pads_short_and_crops_long():
    assert len(F.fix_length(np.ones(100, dtype=np.float32))) == F.N_SAMPLES
    assert len(F.fix_length(np.ones(F.N_SAMPLES + 5000, dtype=np.float32))) == F.N_SAMPLES


def test_frame_features_have_fixed_shapes(a440):
    _, frames, _ = a440
    assert frames["logmel"].shape == (F.N_MELS, F.N_FRAMES)
    assert frames["chroma"].shape == (12, F.N_FRAMES)


def test_names_match_vector_length():
    vec, names = F.summary_vector(tone(220.0, seconds=5))
    assert len(vec) == len(names) == len(set(names))


def test_a440_chroma_peaks_at_a(a440):
    _, frames, _ = a440
    assert F.PITCH_CLASSES[int(np.argmax(frames["chroma"].mean(axis=1)))] == "A"


def test_sine_centroid_near_its_frequency(a440):
    _, _, feats = a440
    assert feats["centroid_mean"] == pytest.approx(440.0, rel=0.05)


def test_noise_is_brighter_and_flatter_than_sine(a440):
    _, _, sine = a440
    rng = np.random.default_rng(0)
    vec, names = F.summary_vector(rng.uniform(-0.5, 0.5, F.N_SAMPLES).astype(np.float32))
    noise = dict(zip(names, vec))
    assert noise["centroid_mean"] > 5 * sine["centroid_mean"]
    assert noise["flatness_mean"] > 10 * sine["flatness_mean"]
    assert noise["zcr_mean"] > 5 * sine["zcr_mean"]


def test_click_track_tempo():
    vec, names = F.summary_vector(clicks(120))
    assert dict(zip(names, vec))["tempo_bpm"] == pytest.approx(120.0, rel=0.03)


def test_silence_gives_finite_features():
    vec, _ = F.summary_vector(np.zeros(F.N_SAMPLES, dtype=np.float32))
    assert np.all(np.isfinite(vec))


def test_frames_match_librosa_frame_count():
    assert F.frames_of(np.zeros(F.N_SAMPLES)).shape == (F.N_FRAMES, F.N_FFT)


def test_framewise_rms_of_sine_is_amp_over_root2(a440):
    _, _, feats = a440
    # edge frames are half zero-padding, so allow a little slack on the mean
    assert feats["rms_mean"] == pytest.approx(0.5 / np.sqrt(2), rel=0.01)


def test_framewise_zcr_of_sine_is_2f_over_sr(a440):
    _, _, feats = a440
    assert feats["zcr_mean"] == pytest.approx(2 * 440 / F.SR, rel=0.02)


def test_framewise_uses_clip_sample_rate(a440):
    # cf.spectral_centroid defaults to sr=44100; at 22050 Hz that would double it
    _, _, feats = a440
    assert feats["centroid_mean"] < 600
