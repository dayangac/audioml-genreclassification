"""GTZAN features: log-mel and chroma per frame, plus a clip-level summary vector."""

import numpy as np
import librosa

import course_features as cf

SR = 22050
DURATION_S = 30.0
N_SAMPLES = int(SR * DURATION_S)
N_FFT = 2048
HOP = 512
N_MELS = 128
N_MFCC = 20
N_FRAMES = 1 + N_SAMPLES // HOP

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_clip(path):
    """Load a clip as mono float32 at SR, cut or padded to N_SAMPLES."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return fix_length(y)


def fix_length(y):
    return librosa.util.fix_length(y, size=N_SAMPLES)


def frame_features(y):
    """Frame-level matrices for neural models, each with N_FRAMES columns."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    return {"logmel": logmel.astype(np.float32), "chroma": chroma.astype(np.float32)}


def frames_of(y):
    """Centred frames, shape (N_FRAMES, N_FFT), matching librosa so every descriptor has the same T."""
    padded = np.pad(y, N_FFT // 2)
    return np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]


def framewise(fn, y, window=None, **kwargs):
    """Apply a whole-signal feature function to each frame; returns (1, T)."""
    frames = frames_of(y)
    if window is not None:
        frames = frames * window
    return np.array([[fn(frame, **kwargs) for frame in frames]])


HANN = np.hanning(N_FFT)  # without it, leakage puts a 440 Hz sine's centroid near 1.1 kHz


def _descriptors(y, logmel, chroma):
    """Per-frame descriptor groups, each a (dims, T) array, plus their names."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    groups = [
        ("mfcc", mfcc, [f"mfcc{i}" for i in range(N_MFCC)]),
        ("dmfcc", librosa.feature.delta(mfcc), [f"dmfcc{i}" for i in range(N_MFCC)]),
        ("chroma", chroma, PITCH_CLASSES),
        ("tonnetz", librosa.feature.tonnetz(chroma=chroma),
         ["fifth_x", "fifth_y", "minor3_x", "minor3_y", "major3_x", "major3_y"]),
        ("centroid", framewise(cf.spectral_centroid, y, window=HANN, sr=SR), [""]),
        ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR), [""]),
        ("rolloff", librosa.feature.spectral_rolloff(S=S, sr=SR), [""]),
        ("flatness", librosa.feature.spectral_flatness(S=S), [""]),
        ("contrast", librosa.feature.spectral_contrast(S=S, sr=SR),
         [f"band{i}" for i in range(7)]),
        ("zcr", framewise(cf.zero_crossing_rate, y), [""]),
        ("rms", framewise(cf.root_mean_square, y), [""]),
        ("onset", onset_env[np.newaxis, :], [""]),
    ]
    return groups, onset_env


def summary_vector(y, frames=None):
    """Clip-level feature vector and matching names (mean/std over time + tempo)."""
    if frames is None:
        frames = frame_features(y)
    groups, onset_env = _descriptors(y, frames["logmel"], frames["chroma"])
    values, names = [], []
    for group, X, dims in groups:
        for stat, fn in (("mean", np.mean), ("std", np.std)):
            values.append(fn(X, axis=1))
            names += [f"{group}{'_' + d if d else ''}_{stat}" for d in dims]
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    values.append(np.atleast_1d(tempo)[:1])
    names.append("tempo_bpm")
    vec = np.concatenate(values).astype(np.float32)
    return np.nan_to_num(vec), names


def extract(path):
    """All features for one audio file."""
    y = load_clip(path)
    frames = frame_features(y)
    vec, names = summary_vector(y, frames)
    return {**frames, "summary": vec, "names": names}
