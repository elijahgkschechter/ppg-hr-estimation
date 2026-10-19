"""The Step 8 model and its finishing touch."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def make_booster(n_trees=300):
    """A fresh, untrained gradient-boosting model that aims to minimize absolute error.

    early_stopping=False matters: scikit-learn's automatic early stopping holds out a
    RANDOM 10% of windows to decide when to stop -- and random windows have overlapping
    twins in training (the Step 7 leak). We choose the number of trees honestly instead.
    """
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=n_trees,
                                         learning_rate=0.1, early_stopping=False,
                                         random_state=0)


def snap_to_peaks(predictions, peak_bpm, max_distance=10):
    """Move each prediction onto the nearest real PPG peak, if one is within max_distance bpm."""
    distance = np.abs(peak_bpm - np.asarray(predictions)[:, None])
    nearest = distance.argmin(axis=1)
    rows = np.arange(len(predictions))
    close_enough = distance[rows, nearest] <= max_distance
    return np.where(close_enough, peak_bpm[rows, nearest], predictions)
