import numpy as np
import librosa


def load_audio(file_path, sr=16000, duration=3):
    y, sr = librosa.load(file_path, sr=sr)

    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y, sr


def extract_mfcc_features(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ])

    return features


def extract_features_from_file(file_path, sr=16000, duration=3, n_mfcc=13):
    y, sr = load_audio(file_path, sr=sr, duration=duration)
    return extract_mfcc_features(y, sr, n_mfcc=n_mfcc)