"""How we score heart-rate estimates."""

import numpy as np


def mae(estimate, truth):
    """Mean absolute error: on average, how many bpm off are the estimates?"""
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def within(estimate, truth, bpm=5):
    """Percent of windows where the estimate is within `bpm` of the truth."""
    return float(100 * np.mean(np.abs(np.asarray(estimate) - np.asarray(truth)) <= bpm))