"""Step 9b: is the CNN result stable, or did we get lucky?

Neural networks start from random numbers, so two training runs never give exactly the same
model. This script trains the same CNN two more times with different random seeds (same
settings, same leave-one-subject-out), then reports the spread across the 3 runs and the
result of averaging the 3 networks' predictions (an "ensemble"; Deep PPG also used one).

Run after scripts/run_step9.py:   python scripts/run_step9_seeds.py   (about 5 minutes)
"""

import json

import numpy as np

from ppghr.data import PROCESSED_DIR, PROJECT_DIR
from ppghr.deep import (best_device, history_index, network_spectra,
                        predict_network, train_network)
from ppghr.evaluate import per_subject_mae
from ppghr.windows import load_all_windows

device = best_device()
data = load_all_windows()
y = data["hr"].astype(np.float32)
subjects = data["subject"]
spectra, bpm_grid = network_spectra(data)
hist = history_index(subjects)

runs = [np.load(PROCESSED_DIR.parent / "step9_predictions.npz")["cnn"]]   # run 1 = Step 9
for run in (2, 3):
    cache_dir = PROCESSED_DIR.parent / "cache" / f"step9_run{run}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.zeros(len(y))
    for subject in np.unique(subjects):
        cache_file = cache_dir / f"S{subject}.npy"
        test_rows = np.where(subjects == subject)[0]
        if not cache_file.exists():
            train_rows = np.where(subjects != subject)[0][::2]
            model = train_network(spectra, hist, y, bpm_grid, train_rows,
                                  seed=1000 * run + int(subject), device=device)
            estimate, _ = predict_network(model, spectra, hist, bpm_grid, test_rows, device=device)
            np.save(cache_file, estimate)
        predictions[test_rows] = np.load(cache_file)
    runs.append(predictions)
    print(f"Run {run}: {per_subject_mae(predictions, y, subjects)[1]:.2f} bpm", flush=True)

scores = [per_subject_mae(p, y, subjects)[1] for p in runs]
ensemble = np.mean(runs, axis=0)
ensemble_per_subject, ensemble_score = per_subject_mae(ensemble, y, subjects)
print(f"\nSingle CNN, 3 runs: {np.mean(scores):.2f} +/- {np.std(scores, ddof=1):.2f} bpm "
      f"(runs: {', '.join(f'{s:.2f}' for s in scores)})")
print(f"Average of the 3 networks (ensemble): {ensemble_score:.2f} bpm "
      f"({100 * np.mean(np.abs(ensemble - y) <= 5):.1f}% within 5 bpm)")

with open(PROJECT_DIR / "results" / "step9b_seeds.json", "w") as f:
    json.dump({"single_run_mae": [round(s, 2) for s in scores],
               "mean": round(float(np.mean(scores)), 2),
               "std": round(float(np.std(scores, ddof=1)), 2),
               "ensemble_of_3_mae": round(ensemble_score, 2),
               "ensemble_per_subject": {f"S{s}": round(v, 2) for s, v in ensemble_per_subject.items()}},
              f, indent=2)
np.save(PROCESSED_DIR.parent / "step9_ensemble_predictions.npy", ensemble)
print("Saved results/step9b_seeds.json")
