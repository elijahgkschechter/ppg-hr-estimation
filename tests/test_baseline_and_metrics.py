import numpy as np
import pytest

from ppghr.baseline import bandpass, estimate_hr_fft
from ppghr.data import FS_PPG
from ppghr.metrics import mae, within

T = np.arange(8 * FS_PPG) / FS_PPG   # one 8-second window of time stamps


@pytest.mark.parametrize("bpm", [45, 72, 150, 200])
def test_fft_finds_the_rate_of_a_clean_pulse(bpm):
    ppg = np.sin(2 * np.pi * bpm / 60 * T)[None, :]
    assert abs(estimate_hr_fft(ppg)[0] - bpm) < 1.0


def test_fft_ignores_slow_drift_outside_the_heart_range():
    drift = 5 * np.sin(2 * np.pi * 0.1 * T)            # 6 "bpm": far below any heart rate
    pulse = np.sin(2 * np.pi * 90 / 60 * T)
    assert abs(estimate_hr_fft((drift + pulse)[None, :])[0] - 90) < 1.0


def test_bandpass_removes_a_constant_offset():
    filtered = bandpass((np.sin(2 * np.pi * 1.2 * T) + 100)[None, :], FS_PPG)
    assert abs(filtered.mean()) < 0.05


def test_mae_and_within():
    assert mae([100, 110, 120], [100, 112, 115]) == pytest.approx((0 + 2 + 5) / 3)
    assert within([100, 110], [103, 100], bpm=5) == 50.0
