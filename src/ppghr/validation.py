"""Step 10: sensor-validation statistics and plots for heart-rate estimates."""

import numpy as np
from scipy import stats


def agreement(estimate, truth):
    """The standard numbers a validation report gives for a heart-rate sensor vs. a reference (ECG)."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    diff = estimate - truth
    abs_err = np.abs(diff)
    bias, sd = diff.mean(), diff.std(ddof=1)
    return {
        "MAE (bpm)": abs_err.mean(),
        "RMSE (bpm)": np.sqrt(np.mean(diff ** 2)),
        "MAPE (%)": 100 * np.mean(abs_err / truth),
        "bias (bpm)": bias,
        "lower limit of agreement (bpm)": bias - 1.96 * sd,
        "upper limit of agreement (bpm)": bias + 1.96 * sd,
        "95th percentile error (bpm)": np.percentile(abs_err, 95),
        "Pearson r": stats.pearsonr(estimate, truth)[0],
        "within 5 bpm (%)": 100 * np.mean(abs_err <= 5),
        "within 5 bpm or 10% (%)": 100 * np.mean(abs_err <= np.maximum(5, 0.1 * truth)),
    }


def bland_altman(ax, estimate, truth, title=""):
    """Bland-Altman plot: x = average of the two methods, y = their difference.
    Lines show the average difference (bias) and the 95% limits of agreement."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    mean, diff = (estimate + truth) / 2, estimate - truth
    bias, sd = diff.mean(), diff.std(ddof=1)
    ax.hexbin(mean, diff, gridsize=80, bins="log", cmap="Blues", mincnt=1)
    for value, style, label in [(bias, "-", f"bias {bias:+.1f}"),
                                (bias + 1.96 * sd, "--", f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                                (bias - 1.96 * sd, "--", f"-1.96 SD {bias - 1.96 * sd:+.1f}")]:
        ax.axhline(value, color="crimson", linestyle=style, linewidth=1, label=label)
    ax.set_xlabel("Average of estimate and ECG (bpm)")
    ax.set_ylabel("Estimate minus ECG (bpm)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def classify_big_errors(estimate, truth, motion_bpm, big=20, tolerance=5):
    """Label each window's error. Big errors (> `big` bpm) are split into:
    - 'half or double the true HR'  (the estimate locked onto a harmonic of the heartbeat)
    - 'locked onto a motion rhythm' (the estimate sits on one of the motion peaks)
    - 'other'"""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    label = np.full(len(estimate), "small error (<= 20 bpm)", dtype=object)
    large = np.abs(estimate - truth) > big
    harmonic = large & ((np.abs(estimate - 2 * truth) <= tolerance)
                        | (np.abs(estimate - truth / 2) <= tolerance))
    near_motion = np.abs(estimate[:, None] - motion_bpm).min(axis=1) <= tolerance
    motion = large & ~harmonic & near_motion
    label[large] = "other big error"
    label[harmonic] = "half or double the true HR"
    label[motion] = "locked onto a motion rhythm"
    return label
