import numpy as np
import pytest

from ppghr.data import FS_ACC, FS_PPG, PROCESSED_DIR, SUBJECTS, check_subject, load_processed
from ppghr.windows import load_all_windows, make_windows

needs_data = pytest.mark.skipif(not (PROCESSED_DIR / "S1.npz").exists(),
                                reason="processed PPG-DaLiA files not available")


def test_check_subject_accepts_a_consistent_recording(fake_subject):
    check_subject(fake_subject())


def test_check_subject_catches_a_short_accelerometer(fake_subject):
    arrays = fake_subject()
    arrays["acc"] = arrays["acc"][: -10 * FS_ACC]
    with pytest.raises(AssertionError, match="accelerometer"):
        check_subject(arrays)


def test_check_subject_catches_a_missing_label(fake_subject):
    arrays = fake_subject()
    arrays["hr"] = arrays["hr"][:-1]
    with pytest.raises(AssertionError, match="heart-rate labels"):
        check_subject(arrays)


def test_windows_line_up_with_labels(fake_subject):
    arrays = fake_subject(seconds=40)
    w = make_windows(arrays)
    n = len(arrays["hr"])
    assert w["ppg"].shape == (n, 8 * FS_PPG)
    assert w["acc"].shape == (n, 8 * FS_ACC, 3)
    i = 5  # window 5 starts at 10 s
    np.testing.assert_array_equal(w["ppg"][i], arrays["ppg"][10 * FS_PPG : 18 * FS_PPG])
    np.testing.assert_array_equal(w["acc"][i], arrays["acc"][10 * FS_ACC : 18 * FS_ACC])


def test_activity_is_read_at_the_window_centre(fake_subject):
    arrays = fake_subject(seconds=40)
    arrays["activity"][:] = 1
    arrays["activity"][20 * 4 :] = 2          # activity changes at second 20
    w = make_windows(arrays)
    # window i is centred at 2*i + 4 seconds, so windows 0-7 are before the change, 8+ after
    assert (w["activity"][:8] == 1).all()
    assert (w["activity"][8:] == 2).all()


@needs_data
def test_every_real_subject_passes_the_checks():
    total = 0
    for subject in SUBJECTS:
        arrays = load_processed(subject)
        check_subject(arrays)
        total += len(arrays["hr"])
    assert total == 64697


@needs_data
def test_all_windows_load_with_the_expected_shapes():
    w = load_all_windows()
    assert w["ppg"].shape == (64697, 512)
    assert w["acc"].shape == (64697, 256, 3)
    assert set(np.unique(w["subject"])) == set(range(1, 16))
