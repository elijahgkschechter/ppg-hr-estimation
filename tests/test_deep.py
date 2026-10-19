import numpy as np
import torch

from conftest import sine_windows
from ppghr.deep import (SpectrumCNN, history_index, network_spectra,
                        predict_network, train_network)


def test_history_index_stays_inside_each_person():
    hist = history_index(np.array([1, 1, 1, 2, 2]), n_history=3)
    np.testing.assert_array_equal(hist, [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 3, 3], [4, 3, 3]])


def test_network_spectra_grid_and_peak():
    spectra, grid = network_spectra(sine_windows(ppg_bpm=90))
    assert spectra.shape == (4, 4, 113)
    assert grid[0] == 30 and grid[-1] == 240
    assert abs(grid[spectra[0, 0].argmax()] - 90) <= 1.875


def test_cnn_outputs_one_score_per_bin():
    model = SpectrumCNN(n_bins=113)
    assert model(torch.zeros(2, 32, 113)).shape == (2, 113)


def test_a_tiny_network_learns_clean_pulses():
    rng = np.random.default_rng(0)
    rates = rng.uniform(60, 160, 300).astype(np.float32)
    parts = [sine_windows(ppg_bpm=r, acc_bpm=rng.uniform(40, 200), n=1) for r in rates]
    windows = {k: np.concatenate([p[k] for p in parts]) for k in ("ppg", "acc")}
    spectra, grid = network_spectra(windows)
    hist = history_index(np.arange(len(rates)))          # every window its own "person"
    rows = np.arange(len(rates))
    device = torch.device("cpu")
    model = train_network(spectra, hist, rates, grid, rows, epochs=15, device=device)
    estimate, _ = predict_network(model, spectra, hist, grid, rows, device=device)
    assert np.mean(np.abs(estimate - rates)) < 5
