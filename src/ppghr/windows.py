"""Cut the continuous signals into 8-second windows that line up with the heart-rate labels."""

import numpy as np

from ppghr.data import (FS_ACC, FS_ACTIVITY, FS_PPG, SHIFT_S, SUBJECTS, WINDOW_S,
                        load_processed)


def make_windows(arrays):
    """Return one row per heart-rate label: the PPG, motion and activity in that window."""
    ppg_rows, acc_rows, activity = [], [], []
    for i in range(len(arrays["hr"])):
        start_s = i * SHIFT_S
        center_s = start_s + WINDOW_S / 2

        p0 = start_s * FS_PPG
        ppg_rows.append(arrays["ppg"][p0 : p0 + WINDOW_S * FS_PPG])

        a0 = start_s * FS_ACC
        acc_rows.append(arrays["acc"][a0 : a0 + WINDOW_S * FS_ACC])

        activity.append(arrays["activity"][int(center_s * FS_ACTIVITY)])

    return {
        "ppg": np.stack(ppg_rows),         # (number of windows, 512)
        "acc": np.stack(acc_rows),         # (number of windows, 256, 3)
        "hr": arrays["hr"],                # (number of windows,)
        "activity": np.array(activity),    # (number of windows,)
    }


def load_all_windows(subjects=SUBJECTS):
    """Window every subject and stack them into one big dataset."""
    parts = []
    for subject in subjects:
        windows = make_windows(load_processed(subject))
        windows["subject"] = np.full(len(windows["hr"]), int(subject[1:]))
        parts.append(windows)
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}