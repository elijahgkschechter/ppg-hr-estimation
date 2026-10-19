import numpy as np

from conftest import sine_windows
from ppghr.features import past_values, peak_features, running_mean, spectral_features
from ppghr.models import make_booster, snap_to_peaks


def test_past_values_never_borrow_from_another_person():
    features = np.arange(6, dtype=float)[:, None]
    subjects = np.array([1, 1, 1, 2, 2, 2])
    out = past_values(features, subjects, n_lags=2)
    np.testing.assert_array_equal(out[:, 0], [0, 0, 1, 3, 3, 4])   # previous window
    np.testing.assert_array_equal(out[:, 1], [0, 0, 0, 3, 3, 3])   # two windows back


def test_running_mean_uses_only_the_past_of_the_same_person():
    features = np.array([[1.0], [2.0], [3.0], [10.0]])
    subjects = np.array([1, 1, 1, 2])
    np.testing.assert_allclose(running_mean(features, subjects, n=2)[:, 0], [1, 1.5, 2.5, 10])


def test_peak_features_find_the_pulse_and_the_motion_rhythm():
    peaks = peak_features(sine_windows(ppg_bpm=84, acc_bpm=120))
    assert peaks.shape == (4, 19)
    assert np.all(np.abs(peaks[:, 0] - 84) < 1.0)     # tallest PPG peak
    assert np.all(np.abs(peaks[:, 10] - 120) < 1.0)   # tallest motion peak


def test_spectral_features_shape():
    assert spectral_features(sine_windows(ppg_bpm=70)).shape == (4, 228)


def test_snap_only_moves_to_a_nearby_peak():
    predictions = np.array([70.0, 100.0])
    peaks = np.array([[72, 150, 40, 41, 42], [130, 140, 150, 160, 170]], dtype=float)
    np.testing.assert_array_equal(snap_to_peaks(predictions, peaks, max_distance=10), [72, 100])


def test_booster_does_not_use_leaky_automatic_early_stopping():
    assert make_booster().early_stopping is False
    assert make_booster(n_trees=1000).max_iter == 1000
