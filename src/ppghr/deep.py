"""Step 9: a small convolutional neural network that looks at the spectra
and points at the frequency bin where the heartbeat is."""

import numpy as np
import torch
from torch import nn

from ppghr.baseline import HIGH_HZ, LOW_HZ, bandpass, spectrum_bpm
from ppghr.data import FS_ACC, FS_PPG

N_FFT = {FS_PPG: 2048, FS_ACC: 1024}   # both give one value every 1.875 bpm
N_HISTORY = 8                          # the current window + the 7 before it
LABEL_WIDTH_BPM = 3.0                  # how "blurry" the training answer is


def network_spectra(windows, batch_size=2000):
    """Return (spectra, bpm_grid). spectra has shape (windows, 4, bins):
    the PPG, x, y and z spectra from 30 to 240 bpm, each scaled so its max is 1."""
    chunks = []
    for start in range(0, len(windows["ppg"]), batch_size):
        ppg = windows["ppg"][start : start + batch_size][:, None, :]
        acc = windows["acc"][start : start + batch_size].transpose(0, 2, 1)
        parts = []
        for signals, fs in [(ppg, FS_PPG), (acc, FS_ACC)]:
            bpm, power = spectrum_bpm(bandpass(signals, fs), fs, N_FFT[fs])
            keep = (bpm >= LOW_HZ * 60) & (bpm <= HIGH_HZ * 60)
            power = power[..., keep]
            parts.append(power / (power.max(axis=-1, keepdims=True) + 1e-12))
        chunks.append(np.concatenate(parts, axis=1).astype(np.float32))
    return np.concatenate(chunks), bpm[keep].astype(np.float32)


def history_index(subjects, n_history=N_HISTORY):
    """For each window: its own row number and the row numbers of the windows just before it
    (same person only; at the start of a recording the first window is repeated)."""
    rows = np.arange(len(subjects))
    first = np.zeros(len(subjects), dtype=int)
    for s in np.unique(subjects):
        where = np.where(subjects == s)[0]
        first[where] = where[0]
    return np.maximum(rows[:, None] - np.arange(n_history)[None, :], first[:, None])


class SpectrumCNN(nn.Module):
    """Input: (batch, 4 signals x N_HISTORY windows, bins). Output: one score per bin."""

    def __init__(self, n_bins, n_history=N_HISTORY, width=32):
        super().__init__()

        def block(c_in, c_out, dilation):
            return [nn.Conv1d(c_in, c_out, kernel_size=7, padding=3 * dilation, dilation=dilation),
                    nn.BatchNorm1d(c_out), nn.ReLU()]

        self.layers = nn.Sequential(
            *block(4 * n_history, width, 1),
            *block(width, width, 1),
            *block(width, width, 2),
            *block(width, width, 4),
            *block(width, width, 8),
            nn.Conv1d(width, 1, kernel_size=1),
        )
        self.prior = nn.Parameter(torch.zeros(n_bins))   # learns which heart rates are common

    def forward(self, x):
        return self.layers(x).squeeze(1) + self.prior


def best_device():
    """Use the Mac's GPU (MPS) when available, otherwise the CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _batch(spectra, hist, rows):
    x = spectra[torch.from_numpy(hist[rows])]          # (batch, history, 4, bins)
    return x.reshape(len(rows), -1, spectra.shape[-1])  # (batch, history*4, bins)


def train_network(spectra, hist, hr, bpm_grid, train_rows, epochs=20, batch_size=256,
                  learning_rate=1e-3, seed=0, device=None):
    """Train a fresh SpectrumCNN on the given rows and return it."""
    device = device or best_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    S = torch.from_numpy(spectra)
    grid = torch.from_numpy(bpm_grid)
    model = SpectrumCNN(len(bpm_grid)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_rows)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = _batch(S, hist, rows).to(device)
            truth = torch.from_numpy(hr[rows])[:, None]
            target = torch.exp(-0.5 * ((grid[None, :] - truth) / LABEL_WIDTH_BPM) ** 2)
            target = (target / target.sum(dim=1, keepdim=True)).to(device)
            loss = -(target * torch.log_softmax(model(x), dim=1)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
    return model


@torch.no_grad()
def predict_network(model, spectra, hist, bpm_grid, rows, batch_size=1024, device=None):
    """Heart-rate estimate for each row: the most likely bin, refined by a weighted
    average of its neighbours. Also returns the full probability for every bin."""
    device = device or best_device()
    model.eval()
    S = torch.from_numpy(spectra)
    probs = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        probs.append(torch.softmax(model(_batch(S, hist, chunk).to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    best = probs.argmax(axis=1)
    near = np.clip(best[:, None] + np.arange(-3, 4)[None, :], 0, len(bpm_grid) - 1)
    weights = np.take_along_axis(probs, near, axis=1)
    estimate = (weights * bpm_grid[near]).sum(axis=1) / weights.sum(axis=1)
    return estimate, probs
