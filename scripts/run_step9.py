"""Step 9: train the neural network with leave-one-subject-out and save the results.

Run from the project folder:   python scripts/run_step9.py

About 20-40 minutes on a Mac. Safe to stop and re-run: each finished person is
saved in data/cache/step9/ and skipped next time.

Needs data/step8_predictions.npz from Step 8 (for the combined model).

Outputs
  results/step9_summary.json    small summary (tracked by git)
  data/step9_predictions.npz    every window's prediction (for Step 10)
"""

import json
import time

import numpy as np

from ppghr.data import ACTIVITY_NAMES, PROCESSED_DIR, PROJECT_DIR
from ppghr.deep import (best_device, history_index, network_spectra,
                        predict_network, train_network)
from ppghr.evaluate import per_subject_mae
from ppghr.features import peak_features
from ppghr.models import snap_to_peaks
from ppghr.windows import load_all_windows

CACHE_DIR = PROCESSED_DIR.parent / "cache" / "step9"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
device = best_device()
print("Training on:", device, flush=True)

data = load_all_windows()
y = data["hr"].astype(np.float32)
subjects, activity = data["subject"], data["activity"]
spectra, bpm_grid = network_spectra(data)
hist = history_index(subjects)
peak_bpm = peak_features(data)[:, :5]
print(f"Network input per window: {hist.shape[1]} windows x {spectra.shape[1]} signals x "
      f"{spectra.shape[2]} bins", flush=True)

# ---- Leave-one-subject-out: a brand-new network for every test person ----
cnn = np.zeros(len(y))
for subject in np.unique(subjects):
    cache_file = CACHE_DIR / f"S{subject}.npy"
    test_rows = np.where(subjects == subject)[0]
    if cache_file.exists():
        cnn[test_rows] = np.load(cache_file)
        print(f"S{subject}: loaded from cache", flush=True)
        continue
    start = time.time()
    # every 2nd window is enough for training: neighbours share 6 of their 8 seconds
    train_rows = np.where(subjects != subject)[0][::2]
    model = train_network(spectra, hist, y, bpm_grid, train_rows, seed=int(subject), device=device)
    cnn[test_rows], _ = predict_network(model, spectra, hist, bpm_grid, test_rows, device=device)
    np.save(cache_file, cnn[test_rows])
    print(f"S{subject}: MAE {np.abs(cnn[test_rows] - y[test_rows]).mean():5.2f} bpm  "
          f"({time.time() - start:.0f} s)", flush=True)

# ---- Compare, and combine with the Step 8 booster (rule fixed in advance) ----
booster = np.load(PROCESSED_DIR.parent / "step8_predictions.npz")["predictions"]
results = {
    "CNN": cnn,
    "CNN, snapped": snap_to_peaks(cnn, peak_bpm),
    "Step 8 booster": booster,
    "CNN + booster average, snapped": snap_to_peaks((cnn + booster) / 2, peak_bpm),
}
summary = {}
print()
for name, p in results.items():
    per_subject, average = per_subject_mae(p, y, subjects)
    errors = np.abs(p - y)
    summary[name] = {
        "mae": round(average, 2),
        "within_5_bpm_percent": round(float(100 * np.mean(errors <= 5)), 1),
        "per_subject": {f"S{s}": round(v, 2) for s, v in per_subject.items()},
        "per_activity": {ACTIVITY_NAMES[a]: round(float(errors[activity == a].mean()), 2)
                         for a in np.unique(activity)},
    }
    print(f"{name:<32} {average:5.2f} bpm   ({summary[name]['within_5_bpm_percent']}% within 5 bpm)")

(PROJECT_DIR / "results").mkdir(exist_ok=True)
with open(PROJECT_DIR / "results" / "step9_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
np.savez(PROCESSED_DIR.parent / "step9_predictions.npz", cnn=cnn,
         combined=results["CNN + booster average, snapped"],
         hr=y, subject=subjects, activity=activity)
print("\nSaved results/step9_summary.json and data/step9_predictions.npz")
