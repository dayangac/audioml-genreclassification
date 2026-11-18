import numpy as np

def zero_crossing_rate(x):
    """Zero-crossing rate of a 1D signal, in [0, 1]."""
    x = np.asarray(x, dtype=float)

    x_sign = np.where(x>0, 1, 0)
    x_diff = np.diff(x_sign)
    return np.mean(np.abs(x_diff))

def root_mean_square(x):
    """RMS of a 1D signal."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x**2))


def spectral_centroid(x, sr=44100):
    """Spectral centroid of the whole signal, in Hz."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)

    X_abs = np.abs(X)

    denom = X_abs.sum()

    if denom <= 1e-12:  # silent input
        return 0.0 
    
    centroid_hz = (f * X_abs).sum() / denom
    return centroid_hz
