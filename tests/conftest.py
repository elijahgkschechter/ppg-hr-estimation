"""Shared test helpers: small, made-up recordings whose correct answers we know exactly."""

import numpy as np
import pytest

from ppghr.data import FS_ACC, FS_ACTIVITY, FS_PPG, SHIFT_S, WINDOW_S


@pytest.fixture
def fake_subject():
    """Build a fake recording shaped exactly like a load_processed() result.

    The PPG is a perfect sine wave at `bpm` beats per minute, so the true heart rate is known.
    """
    def make(seconds=60, bpm=72.0, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(seconds * FS_PPG) / FS_PPG
        n_labels = int((seconds - WINDOW_S) // SHIFT_S) + 1
        return {
            "ppg": np.sin(2 * np.pi * bpm / 60 * t).astype(np.float32),
            "acc": (0.01 * rng.standard_normal((seconds * FS_ACC, 3))).astype(np.float32),
            "hr": np.full(n_labels, bpm, dtype=np.float32),
            "activity": np.ones(seconds * FS_ACTIVITY, dtype=np.int8),
            "age": 30, "gender": "f", "height_cm": 170.0, "weight_kg": 60.0,
            "skin_type": 3, "sport": 5,
        }
    return make


def sine_windows(ppg_bpm, acc_bpm=None, n=4):
    """`n` identical 8-second windows: PPG sine at ppg_bpm; optional motion sine (x axis) at acc_bpm."""
    t_ppg = np.arange(WINDOW_S * FS_PPG) / FS_PPG
    t_acc = np.arange(WINDOW_S * FS_ACC) / FS_ACC
    ppg = np.tile(np.sin(2 * np.pi * ppg_bpm / 60 * t_ppg), (n, 1)).astype(np.float32)
    acc = np.zeros((n, WINDOW_S * FS_ACC, 3), dtype=np.float32)
    acc[:, :, 2] = 1.0                                   # gravity on the z axis
    if acc_bpm is not None:
        acc[:, :, 0] = np.sin(2 * np.pi * acc_bpm / 60 * t_acc)
    return {"ppg": ppg, "acc": acc, "subject": np.ones(n, dtype=int)}
