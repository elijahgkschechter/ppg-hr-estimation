"""Turn each 8-second window into numbers a model can learn from."""

import numpy as np
from scipy.signal import find_peaks

from ppghr.baseline import HIGH_HZ, LOW_HZ, bandpass, spectrum_bpm

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




# ---- Step 8: richer features ----------------------------------------------

FINE_N_FFT = 4096   # fine spectrum (0.94-bpm steps) to locate peaks precisely


def _top_peaks(power, bpm, k):
    """For each spectrum row: the bpm and relative height (0 to 1) of its k tallest peaks."""
    where = np.zeros((len(power), k), dtype=np.float32)
    height = np.zeros((len(power), k), dtype=np.float32)
    for i, row in enumerate(power):
        peaks, _ = find_peaks(row)
        if len(peaks) == 0:
            peaks = np.array([row.argmax()])
        tallest = peaks[np.argsort(row[peaks])[::-1][:k]]
        n = len(tallest)
        where[i, :n] = bpm[tallest]
        where[i, n:] = bpm[tallest[-1]]
        height[i, :n] = row[tallest] / row.max()
    return where, height


def peak_features(windows, batch_size=2000):
    """Per window: where the 5 tallest PPG peaks and 3 tallest motion peaks are,
    how tall they are, and how much the wrist moved. 19 numbers."""
    rows = []
    for start in range(0, len(windows["ppg"]), batch_size):
        ppg = windows["ppg"][start : start + batch_size]
        acc = windows["acc"][start : start + batch_size]

        bpm, ppg_power = spectrum_bpm(bandpass(ppg, FS_PPG), FS_PPG, FINE_N_FFT)
        keep = (bpm >= LOW_HZ * 60) & (bpm <= HIGH_HZ * 60)
        ppg_where, ppg_height = _top_peaks(ppg_power[:, keep], bpm[keep], 5)

        bpm_a, acc_power = spectrum_bpm(bandpass(acc.transpose(0, 2, 1), FS_ACC),
                                        FS_ACC, FINE_N_FFT // 2)
        keep_a = (bpm_a >= LOW_HZ * 60) & (bpm_a <= HIGH_HZ * 60)
        acc_where, acc_height = _top_peaks(acc_power[..., keep_a].sum(axis=1), bpm_a[keep_a], 3)

        motion = np.linalg.norm(acc, axis=2)
        rows.append(np.column_stack([ppg_where, ppg_height, acc_where, acc_height,
                                     motion.std(axis=1), motion.mean(axis=1), ppg.std(axis=1)]))
    return np.concatenate(rows).astype(np.float32)


def past_values(features, subjects, n_lags):
    """For each window, copy the features of the previous n_lags windows (same person only).
    Only the PAST is used, so a real watch could do this live."""
    blocks = []
    for lag in range(1, n_lags + 1):
        shifted = np.empty_like(features)
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            f = features[idx]
            shifted[idx] = np.concatenate([np.repeat(f[:1], lag, axis=0), f[:-lag]])
        blocks.append(shifted)
    return np.concatenate(blocks, axis=1)


def running_mean(features, subjects, n):
    """Average of the current window and the n-1 before it (same person only)."""
    out = np.empty_like(features)
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        total = np.cumsum(np.vstack([np.zeros((1, features.shape[1])), features[idx]]), axis=0)
        i = np.arange(len(idx))
        low = np.maximum(0, i - n + 1)
        out[idx] = (total[i + 1] - total[low]) / (i + 1 - low)[:, None]
    return out


def build_features(windows):
    """All Step 8 features in one table, plus the column numbers of each group."""
    subjects = windows["subject"]
    spectra = spectral_features(windows)
    peaks = peak_features(windows)
    n_bins = spectra.shape[1] // 4
    ppg_spectrum = spectra[:, :n_bins]
    blocks = {
        "spectra": spectra,
        "peaks": peaks,
        "peak_history": past_values(peaks, subjects, 15),
        "ppg_average": np.concatenate([running_mean(ppg_spectrum, subjects, 4),
                                       running_mean(ppg_spectrum, subjects, 15)], axis=1),
    }
    columns, start = {}, 0
    for name, block in blocks.items():
        columns[name] = np.arange(start, start + block.shape[1])
        start += block.shape[1]
    X = np.concatenate(list(blocks.values()), axis=1).astype(np.float32)
    return X, columns, peaks[:, :5]