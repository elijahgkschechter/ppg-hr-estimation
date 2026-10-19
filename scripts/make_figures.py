"""Step 12: make the figures shown in the README (saved to figures/).

Run after Steps 8-9:   python scripts/make_figures.py   (about 30 seconds)
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ppghr.baseline import bandpass, estimate_hr_fft, spectrum_bpm
from ppghr.data import ACTIVITY_NAMES, FS_ACC, FS_PPG, PROCESSED_DIR, PROJECT_DIR
from ppghr.evaluate import per_subject_mae
from ppghr.windows import load_all_windows

# ---- Colors: one fixed color per method, used in every figure ----
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FFT_COLOR, BOOSTER_COLOR, CNN_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---- Load everything ----
data = load_all_windows()
truth, subjects = data["hr"].astype(float), data["subject"]
activity = np.array([ACTIVITY_NAMES[a] for a in data["activity"]])
fft = estimate_hr_fft(data["ppg"])
booster = np.load(PROCESSED_DIR.parent / "step8_predictions.npz")["predictions"]
final = np.load(PROCESSED_DIR.parent / "step9_ensemble_predictions.npy")
seeds = json.load(open(PROJECT_DIR / "results" / "step9b_seeds.json"))
step8 = json.load(open(PROJECT_DIR / "results" / "step8_summary.json"))

# ---- 1. Scoreboard ----
scores = [  # (name, MAE, from this project?)
    ("FFT peak-picking (this project)", per_subject_mae(fft, truth, subjects)[1], True),
    ("SpaMa (Reiss et al. 2019)", 15.56, False),
    ("k-nearest neighbours (this project)", 13.1, True),
    ("SpaMaPlus (Reiss et al. 2019)", 11.06, False),
    ("Deep PPG CNN ensemble (Reiss et al. 2019)", 7.65, False),
    ("Gradient boosting, nested CV (this project)", step8["nested_loso_mae"], True),
    ("CNN, single network (this project)", seeds["mean"], True),
    ("CNN ensemble of 3 (this project)", seeds["ensemble_of_3_mae"], True),
]
scores.sort(key=lambda row: row[1], reverse=True)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
y = np.arange(len(scores))
ax.barh(y, [s[1] for s in scores], height=0.62,
        color=[CNN_COLOR if "ensemble of 3" in s[0] else (FFT_COLOR if s[2] else MUTED) for s in scores])
for i, (name, value, _) in enumerate(scores):
    ax.text(value + 0.25, i, f"{value:.2f}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [s[0] for s in scores], color=INK_2)
ax.set_xlabel("Mean absolute error (bpm), leave-one-subject-out; lower is better")
ax.set_xlim(0, 20.5)
ax.grid(axis="y", visible=False)
ax.set_title("Wrist-PPG heart-rate error on PPG-DaLiA (15 people)",
             loc="left", fontsize=11, color=INK)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CNN_COLOR, FFT_COLOR, MUTED)]
ax.legend(handles, ["Final model", "This project", "Published"], loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "scoreboard.png", dpi=160)
plt.close(fig)

# ---- 2. Error by activity (dot plot) ----
methods = [("FFT baseline", fft, FFT_COLOR), ("Gradient boosting", booster, BOOSTER_COLOR),
           ("CNN ensemble (final)", final, CNN_COLOR)]
names = sorted(np.unique(activity), key=lambda a: np.abs(final - truth)[activity == a].mean(), reverse=True)
fig, ax = plt.subplots(figsize=(8, 4.4))
for i, a in enumerate(names):
    values = [np.abs(p - truth)[activity == a].mean() for _, p, _ in methods]
    ax.plot([min(values), max(values)], [i, i], color=GRID, linewidth=2, zorder=1)
    for (label, _, color), v in zip(methods, values):
        ax.scatter(v, i, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=label if i == 0 else None)
    ax.text(values[2] - 0.8, i, f"{values[2]:.1f}", va="center", ha="right", fontsize=8, color=INK_2)
ax.set_yticks(range(len(names)), names, color=INK_2)
ax.set_xlabel("Mean absolute error (bpm)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 41)
ax.set_title("Error by activity: motion is what breaks the classic method", loc="left", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "error_by_activity.png", dpi=160)
plt.close(fig)

# ---- 3. Bland-Altman ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (label, p) in zip(axes, [("FFT baseline", fft), ("CNN ensemble (final model)", final)]):
    mean, diff = (p + truth) / 2, p - truth
    bias, sd = diff.mean(), diff.std(ddof=1)
    ax.hexbin(mean, diff, gridsize=70, bins="log", cmap=BLUES, mincnt=1, linewidths=0)
    for value, style, text in [(bias, "-", f"bias {bias:+.1f}"),
                               (bias + 1.96 * sd, "--", f"+1.96 SD  {bias + 1.96 * sd:+.1f}"),
                               (bias - 1.96 * sd, "--", f"-1.96 SD  {bias - 1.96 * sd:+.1f}")]:
        ax.axhline(value, color=INK, linestyle=style, linewidth=1)
        ax.text(197, value + 2, text, ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(40, 200)
    ax.set_title(label, loc="left", fontsize=11)
    ax.set_xlabel("Average of estimate and ECG (bpm)")
axes[0].set_ylim(-110, 110)
axes[0].set_ylabel("Estimate minus ECG (bpm)")
fig.suptitle("Bland-Altman agreement with chest ECG (every 8-second window; darker = more windows)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "bland_altman.png", dpi=160)
plt.close(fig)

# ---- 4. Why the problem is hard: one stairs window ----
i = 450   # subject S1, minute 15, climbing stairs
bpm, ppg_power = spectrum_bpm(bandpass(data["ppg"][i], FS_PPG), FS_PPG)
bpm_a, acc_power = spectrum_bpm(bandpass(data["acc"][i].T, FS_ACC), FS_ACC)
acc_power = acc_power.sum(axis=0)
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.plot(bpm, ppg_power / ppg_power.max(), color=FFT_COLOR, linewidth=2, label="Wrist PPG spectrum")
ax.plot(bpm_a, acc_power / acc_power.max(), color=BOOSTER_COLOR, linewidth=2, label="Wrist motion spectrum")
ax.axvline(truth[i], color=INK, linestyle="-", linewidth=1.2)
ax.text(truth[i] + 1.5, 1.04, f"true heart rate (ECG): {truth[i]:.0f}", fontsize=8, color=INK)
ax.axvline(fft[i], color=INK, linestyle=":", linewidth=1.2)
ax.text(fft[i] - 1.5, 1.04, f"FFT picks the tallest peak: {fft[i]:.0f}", fontsize=8, color=INK, ha="right")
ax.set_xlim(30, 240)
ax.set_ylim(0, 1.15)
ax.set_xlabel("Rhythm speed (bpm)")
ax.set_ylabel("Strength (scaled)")
ax.set_title("Why it's hard: on the stairs, motion swamps the pulse in the wrist PPG (subject S1)",
             loc="left", fontsize=11)
ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(1, 0.88))
fig.tight_layout()
fig.savefig(FIG_DIR / "why_it_is_hard.png", dpi=160)
plt.close(fig)

# ---- 5. Tracking one person through the whole recording ----
rows = subjects == 1
minutes = (np.arange(rows.sum()) * 2 + 4) / 60
fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, sharey=True)
for ax, (label, p, color) in zip(axes, [("FFT baseline", fft, FFT_COLOR),
                                        ("CNN ensemble (final)", final, CNN_COLOR)]):
    acts = activity[rows]
    for k, a in enumerate(["stairs", "table soccer", "cycling", "walking"]):
        on = acts == a
        ax.fill_between(minutes, 30, 200, where=on, color=GRID, alpha=0.8, linewidth=0)
        if ax is axes[0] and on.any():
            ax.text(minutes[on].mean(), 196 - 11 * (k % 2), a, ha="center", va="top",
                    fontsize=8, color=INK_2)
    ax.plot(minutes, p[rows], color=color, linewidth=1, label=label)
    ax.plot(minutes, truth[rows], color=INK, linewidth=1.4, label="ECG (truth)")
    mae = np.abs(p[rows] - truth[rows]).mean()
    ax.set_title(f"{label}: MAE {mae:.1f} bpm", loc="left", fontsize=10)
    ax.set_ylabel("Heart rate (bpm)")
    ax.legend(loc="lower right", fontsize=8, ncol=2, bbox_to_anchor=(1, 1.0))
axes[1].set_xlabel("Minutes into the recording (subject S1)")
axes[0].set_ylim(30, 200)
fig.tight_layout()
fig.savefig(FIG_DIR / "tracking_S1.png", dpi=160)
plt.close(fig)

print("Saved:", ", ".join(sorted(p.name for p in FIG_DIR.glob("*.png"))))
