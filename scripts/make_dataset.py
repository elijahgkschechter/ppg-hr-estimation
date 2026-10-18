"""Convert every PPG-DaLiA subject from a huge .pkl file into a small .npz file.

Run from the project folder:  python scripts/make_dataset.py
Safe to run again: subjects that are already converted are skipped.
"""

from ppghr.data import (FS_PPG, PROCESSED_DIR, SUBJECTS, check_subject,
                        extract_wrist, load_raw_subject, save_processed)

for subject in SUBJECTS:
    if (PROCESSED_DIR / f"{subject}.npz").exists():
        print(f"{subject}: already done, skipping")
        continue

    raw = load_raw_subject(subject)
    arrays = extract_wrist(raw)
    del raw  # free the ~1.5 GB before loading the next subject

    check_subject(arrays)
    path = save_processed(subject, arrays)

    minutes = len(arrays["ppg"]) / FS_PPG / 60
    size_mb = path.stat().st_size / 1e6
    print(f"{subject}: {minutes:5.1f} min, {len(arrays['hr']):4d} heart-rate windows, "
          f"saved {size_mb:4.1f} MB")

print("Finished. Files are in", PROCESSED_DIR)