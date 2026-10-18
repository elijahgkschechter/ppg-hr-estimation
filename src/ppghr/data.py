"""Load PPG-DaLiA subjects and convert them into small, fast files."""

import pickle
from pathlib import Path

import numpy as np

# ---- Where things live ---------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR.parent / "ppg+dalia" / "PPG_FieldStudy"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# ---- Facts about the dataset ---------------------------------------------
SUBJECTS = [f"S{i}" for i in range(1, 16)]
FS_PPG = 64        # wrist PPG samples per second
FS_ACC = 32        # wrist accelerometer samples per second
FS_ACTIVITY = 4    # activity label samples per second
WINDOW_S = 8       # each heart-rate label covers 8 seconds...
SHIFT_S = 2        # ...and a new label starts every 2 seconds
ACTIVITY_NAMES = {0: "transition", 1: "sitting", 2: "stairs", 3: "table soccer",
                  4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}


def load_raw_subject(subject):
    """Open one subject's big .pkl file and return everything in it."""
    path = RAW_DIR / subject / f"{subject}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def extract_wrist(raw):
    """Keep only what a wrist device would have, plus the answer key."""
    quest = raw["questionnaire"]
    return {
        "ppg": raw["signal"]["wrist"]["BVP"][:, 0].astype(np.float32),
        "acc": raw["signal"]["wrist"]["ACC"].astype(np.float32),
        "hr": raw["label"].astype(np.float32),
        "activity": raw["activity"][:, 0].astype(np.int8),
        "age": quest["AGE"],
        "gender": quest["Gender"].strip(),
        "height_cm": quest["HEIGHT"],
        "weight_kg": quest["WEIGHT"],
        "skin_type": quest["SKIN"],
        "sport": quest["SPORT"],
    }


def check_subject(arrays):
    """Stop with an error if the signals don't line up the way the dataset promises."""
    seconds = len(arrays["ppg"]) / FS_PPG
    assert len(arrays["acc"]) / FS_ACC == seconds, "accelerometer length does not match PPG"
    assert len(arrays["activity"]) / FS_ACTIVITY == seconds, "activity length does not match PPG"
    expected_windows = int((seconds - WINDOW_S) // SHIFT_S) + 1
    assert len(arrays["hr"]) == expected_windows, (
        f"expected {expected_windows} heart-rate labels, found {len(arrays['hr'])}")
    assert arrays["acc"].shape[1] == 3, "accelerometer should have 3 columns (x, y, z)"
    assert np.isfinite(arrays["hr"]).all(), "heart-rate labels contain missing values"


def save_processed(subject, arrays):
    """Write one subject's small file to data/processed/ and return its path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{subject}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_processed(subject):
    """Read one subject's small file back as a dictionary of arrays."""
    with np.load(PROCESSED_DIR / f"{subject}.npz") as f:
        return {key: f[key] for key in f.files}