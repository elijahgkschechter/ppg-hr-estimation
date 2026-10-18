"""Non-ML baseline: heart rate = the strongest rhythm in the wrist PPG."""

import numpy as np
from scipy.signal import butter, filtfilt

from ppghr.data import FS_PPG

LOW_HZ = 0.5      # 30 bpm: slowest heart rate we look for
HIGH_HZ = 4.0     # 240 bpm: fastest heart rate we look for
N_FFT = 4096      # pad each window to 4096 points for a smoother spectrum


def bandpass(signals, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ, order=4):
    """Keep only rhythms between low_hz and high_hz; remove slower drift and faster noise."""
    b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs)
    return filtfilt(b, a, signals, axis=-1)


def spectrum_bpm(signals, fs, n_fft=N_FFT):
    """Split each window into its rhythms. Returns (rhythm speeds in bpm, strength of each)."""
    bpm = np.fft.rfftfreq(n_fft, d=1 / fs) * 60
    power = np.abs(np.fft.rfft(signals, n=n_fft, axis=-1)) ** 2
    return bpm, power


def estimate_hr_fft(ppg_windows, batch_size=2000):
    """Estimate heart rate for each PPG window as its strongest rhythm between 30 and 240 bpm."""
    estimates = []
    for start in range(0, len(ppg_windows), batch_size):
        batch = bandpass(ppg_windows[start : start + batch_size], FS_PPG)
        bpm, power = spectrum_bpm(batch, FS_PPG)
        in_range = (bpm >= LOW_HZ * 60) & (bpm <= HIGH_HZ * 60)
        strongest = np.argmax(power[:, in_range], axis=1)
        estimates.append(bpm[in_range][strongest])
    return np.concatenate(estimates)
