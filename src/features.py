"""GTZAN features"""

import numpy as np
import librosa

import feat_ext

# default properties of LibRosa for GTZAN files spelled out for more expressive definition
# sampling rate 
SR = 22050
# Duration
DURATION_S = 30.0
# num of samples
N_SAMPLES = int(SR * DURATION_S)
# number of fast fourier tranforms, default of librosa -- fft turns signal over time into frequencies it contains.
N_FFT = 2048
# how far a frame moves before next fft, about 23 ms
HOP = 512
# num of freq bands
N_MELS = 128
# score spectrums shape, important to distinguish instruments
N_MFCC = 20
## how many slices of a clip, for fft
N_FRAMES = 1 + N_SAMPLES // HOP
# musical notes
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_clip(path):
    # Load a clip as mono float32 at SR, cut or padded to N_SAMPLES.
    y, _ = librosa.load(path, sr=SR, mono=True)
    return fix_length(y)


def fix_length(y):
    # ensure all clips same length
    return librosa.util.fix_length(y, size=N_SAMPLES)


def frame_features(y):
    """
    Log-mel (N_MELS, N_FRAMES) and chroma (12, N_FRAMES) of a clip.
    these help use with getting the timbre via log-mel, how it sounds, and the harmony via chroma
    """
    # this turns waveform into table of how much energy in each freq band 
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    return {"logmel": logmel.astype(np.float32), "chroma": chroma.astype(np.float32)}


def frames_of(y):
    """Frames for ZCR, RMS and centroid, centred like librosa by adding silence to both ends so all features line up in time."""
    padded = np.pad(y, N_FFT // 2)
    return np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]


def framewise(fn, y, window=None, **kwargs):
    """Apply a whole-signal feature function to each frame; returns (1, T)."""
    frames = frames_of(y)
    if window is not None:
        frames = frames * window
    return np.array([[fn(frame, **kwargs) for frame in frames]])


HANN = np.hanning(N_FFT)  # fade frame edges to avoid false frequencies

def _descriptors(y, logmel, chroma):
    """ computes every per-frame feature and packs them into one list so summary_vector can process them in a single loop"""
    # magnitude spectogram
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    # 20 tone colur numbers per frame
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
    # measures how storngly something starts in each moment, e.g. drum hit
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    # list of all features
    groups = [
        ("mfcc", mfcc, [f"mfcc{i}" for i in range(N_MFCC)]),
        ("dmfcc", librosa.feature.delta(mfcc), [f"dmfcc{i}" for i in range(N_MFCC)]),
        ("chroma", chroma, PITCH_CLASSES),
        ("tonnetz", librosa.feature.tonnetz(chroma=chroma),
         ["fifth_x", "fifth_y", "minor3_x", "minor3_y", "major3_x", "major3_y"]),
        ("centroid", framewise(feat_ext.spectral_centroid, y, window=HANN, sr=SR), [""]),
        ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR), [""]),
        ("rolloff", librosa.feature.spectral_rolloff(S=S, sr=SR), [""]),
        ("flatness", librosa.feature.spectral_flatness(S=S), [""]),
        ("contrast", librosa.feature.spectral_contrast(S=S, sr=SR),
         [f"band{i}" for i in range(7)]),
        ("zcr", framewise(feat_ext.zero_crossing_rate, y), [""]),
        ("rms", framewise(feat_ext.root_mean_square, y), [""]),
        ("onset", onset_env[np.newaxis, :], [""]),
    ]
    return groups, onset_env


def summary_vector(y, frames=None):
    """squeeze the clip into numbers."""
    # get the features and frames
    if frames is None:
        frames = frame_features(y)
    groups, onset_env = _descriptors(y, frames["logmel"], frames["chroma"])
    values, names = [], []
    # average the features over time
    for group, X, dims in groups:
        for stat, fn in (("mean", np.mean), ("std", np.std)):
            values.append(fn(X, axis=1))
            # build feature names list               
            names += [f"{group}{'_' + d if d else ''}_{stat}" for d in dims]
    # estimate tempo and add as last value
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    values.append(np.atleast_1d(tempo)[:1])
    names.append("tempo_bpm")
    vec = np.concatenate(values).astype(np.float32)
    # return final names list
    return np.nan_to_num(vec), names


def extract(path):
    """returns a dict per audio file with all features and their values"""
    y = load_clip(path)
    frames = frame_features(y)
    vec, names = summary_vector(y, frames)
    return {**frames, "summary": vec, "names": names}
