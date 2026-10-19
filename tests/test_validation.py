import numpy as np
import pytest

from ppghr.validation import agreement, classify_big_errors


def test_perfect_estimates_agree_perfectly():
    truth = np.array([60.0, 80, 100, 120])
    stats = agreement(truth, truth)
    assert stats["MAE (bpm)"] == 0 and stats["bias (bpm)"] == 0
    assert stats["within 5 bpm (%)"] == 100


def test_a_constant_offset_shows_up_as_bias():
    truth = np.array([60.0, 80, 100, 120])
    stats = agreement(truth + 3, truth)
    assert stats["bias (bpm)"] == pytest.approx(3)
    assert stats["MAE (bpm)"] == pytest.approx(3)
    assert stats["lower limit of agreement (bpm)"] == pytest.approx(3)


def test_big_errors_are_sorted_by_cause():
    truth = np.full(5, 100.0)
    estimate = np.array([200, 50, 70, 140, 105], dtype=float)
    motion = np.array([[70, 0, 0]] * 5, dtype=float)
    labels = classify_big_errors(estimate, truth, motion)
    assert list(labels) == ["half or double the true HR", "half or double the true HR",
                            "locked onto a motion rhythm", "other big error",
                            "small error (<= 20 bpm)"]
