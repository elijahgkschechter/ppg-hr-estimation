"""Turn each 8-second window into numbers a model can learn from."""

import numpy as np

from ppghr.baseline import HIGH_HZ, LOW_HZ, bandpass
from ppghr.data import FS_ACC, FS_PPG

FEATURE_N_FFT = {FS_PPG: 1024, FS_ACC: 512}   # both give one value every 3.75 bpm


def scaled_spectrum(signals, fs):
    """Spectrum between 30 and 240 bpm, each window scaled so its tallest peak = 1."""
    n_fft = FEATURE_N_FFT[fs]
    bpm = np.fft.rfftfreq(n_fft, d=1 / fs) * 60
    keep = (bpm >= LOW_HZ * 60) & (bpm <= HIGH_HZ * 60)
    power = np.abs(np.fft.rfft(bandpass(signals, fs), n=n_fft, axis=-1)) ** 2
    power = power[..., keep]
    return power / (power.max(axis=-1, keepdims=True) + 1e-12)


def spectral_features(windows, batch_size=2000):
    """One row per window: the PPG spectrum, then the x, y and z motion spectra."""
    rows = []
    for start in range(0, len(windows["ppg"]), batch_size):
        ppg = scaled_spectrum(windows["ppg"][start : start + batch_size], FS_PPG)
        acc = windows["acc"][start : start + batch_size].transpose(0, 2, 1)
        acc = scaled_spectrum(acc, FS_ACC)
        rows.append(np.concatenate([ppg, acc.reshape(len(acc), -1)], axis=1))
    return np.concatenate(rows).astype(np.float32)