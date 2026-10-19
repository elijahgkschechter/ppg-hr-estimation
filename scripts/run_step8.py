"""Step 8: run the gradient-boosting experiments and save the results.

Run from the project folder:   python scripts/run_step8.py

Takes about an hour (the nested part trains ~300 models). Safe to stop and re-run:
finished pieces are saved in data/cache/step8/ and skipped next time.

Outputs
  results/step8_summary.json    small summary (tracked by git)
  data/step8_predictions.npz    every window's prediction (for the Step 10 error analysis)
"""

import json

import numpy as np

from ppghr.data import ACTIVITY_NAMES, PROCESSED_DIR, PROJECT_DIR
from ppghr.evaluate import nested_loso, per_subject_mae
from ppghr.features import build_features
from ppghr.models import make_booster, snap_to_peaks
from ppghr.windows import load_all_windows

CACHE_DIR = PROCESSED_DIR.parent / "cache" / "step8"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Building features...", flush=True)
data = load_all_windows()
y, subjects, activity = data["hr"], data["subject"], data["activity"]
X, columns, peak_bpm = build_features(data)
print(f"X: {X.shape}", flush=True)

feature_sets = {
    "spectra + peaks": np.concatenate([columns["spectra"], columns["peaks"]]),
    "+ peak history": np.concatenate([columns["spectra"], columns["peaks"], columns["peak_history"]]),
    "+ averages": np.arange(X.shape[1]),
}

# ---- 1. Exploration: plain leave-one-subject-out, 300 trees ---------------
exploration = {}
for set_number, (name, cols) in enumerate(feature_sets.items()):
    p = np.zeros(len(y))
    for subject in np.unique(subjects):
        cache_file = CACHE_DIR / f"explore_set{set_number}_S{subject}.npy"
        test = subjects == subject
        if cache_file.exists():
            p[test] = np.load(cache_file)
            continue
        model = make_booster(n_trees=300)
        model.fit(X[~test][:, cols], y[~test])
        p[test] = model.predict(X[test][:, cols])
        np.save(cache_file, p[test])
    exploration[name] = {
        "plain": round(per_subject_mae(p, y, subjects)[1], 2),
        "snapped": round(per_subject_mae(snap_to_peaks(p, peak_bpm), y, subjects)[1], 2),
    }
    print(f"Exploration  {name:<16} plain {exploration[name]['plain']:5.2f}  "
          f"snapped {exploration[name]['snapped']:5.2f} bpm", flush=True)

# ---- 2. The honest number: nested leave-one-subject-out -------------------
# inner_folds=3: the 14 training people are split into 3 groups for the inner
# competition (instead of 5) to keep the run time reasonable; still fully honest.
predictions, choices = nested_loso(make_booster, X, y, subjects, feature_sets, peak_bpm,
                                   tree_options=(300, 1000), inner_folds=3,
                                   cache_dir=CACHE_DIR)
per_subject, average = per_subject_mae(predictions, y, subjects)
errors = np.abs(predictions - y)
per_activity = {ACTIVITY_NAMES[a]: round(float(errors[activity == a].mean()), 2)
                for a in np.unique(activity)}

summary = {
    "exploration_loso_300_trees": exploration,
    "nested_loso_mae": round(average, 2),
    "nested_per_subject_mae": {f"S{s}": round(v, 2) for s, v in per_subject.items()},
    "nested_choices": {f"S{s}": {"features": c[0], "trees": int(c[1]),
                                 "snap": None if c[2] is None else int(c[2])}
                       for s, c in choices.items()},
    "nested_per_activity_mae": per_activity,
    "within_5_bpm_percent": round(float(100 * np.mean(errors <= 5)), 1),
}
(PROJECT_DIR / "results").mkdir(exist_ok=True)
with open(PROJECT_DIR / "results" / "step8_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
np.savez(PROCESSED_DIR.parent / "step8_predictions.npz",
         predictions=predictions, hr=y, subject=subjects, activity=activity)

print(f"\nHonest (nested) leave-one-subject-out MAE: {average:.2f} bpm")
print("Saved results/step8_summary.json and data/step8_predictions.npz")