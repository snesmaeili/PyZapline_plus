"""
Parity and guard tests for MATLAB zapline-plus behavior.

Covers PSD windowing, small-chunk guards, Nyquist guards, and nt_* fixes.
"""

import numpy as np
import pytest
from scipy import signal
from scipy.signal import windows

from pyzaplineplus import PyZaplinePlus


def test_psd_parity_hamming_peak_bin():
    # Synthetic: 50 Hz line + noise
    fs = 500
    duration = 20.0
    t = np.arange(0, duration, 1 / fs)
    x = 0.2 * np.random.randn(t.size, 4)
    x += 1.5 * np.sin(2 * np.pi * 50 * t)[:, None]

    # Window ~1s for resolution; enforce Hamming + 50% overlap
    win_sec = 1.0
    nperseg = int(win_sec * fs)
    zp = PyZaplinePlus(x, fs, plotResults=False, winSizeCompleteSpectrum=win_sec)
    pxx_log, f = zp.compute_spectrum(x)

    # Reference PSD using periodic Hamming and 50% overlap
    ref_win = windows.hamming(nperseg, sym=False)
    f_ref, pxx_ref = signal.welch(
        x, fs=fs, window=ref_win, nperseg=nperseg, noverlap=nperseg // 2, axis=0
    )

    # Compare peak bin near 50 Hz on mean PSD across channels
    mean_pxx = np.mean(10 * np.log10(pxx_ref), axis=1)
    mean_pxx_zp = np.mean(pxx_log, axis=1)
    band = (f_ref >= 40) & (f_ref <= 60)
    idx_ref = np.argmax(mean_pxx[band])
    idx_zp = np.argmax(mean_pxx_zp[band])
    # Ensure both peak at the same frequency bin
    assert np.allclose(f_ref[band][idx_ref], f[band][idx_zp])


def test_detect_chunk_noise_small_bins_no_crash():
    fs = 1000
    # Very short chunk -> few bins in 6 Hz window
    chunk = np.random.randn(32, 2)
    zp = PyZaplinePlus(chunk, fs, plotResults=False)
    # Use 50 Hz target
    noise_freq = 50.0
    out = zp.detect_chunk_noise(chunk, noise_freq)
    # Should return the original frequency due to <3 bins guard
    assert out == pytest.approx(noise_freq)


def test_bandpass_filter_highcut_above_nyquist_no_crash():
    fs = 500
    x = np.random.randn(1000, 3)
    zp = PyZaplinePlus(x, fs, plotResults=False)
    # highcut above Nyquist: should not crash; either clamp or no-op
    y = zp.bandpass_filter(x, 45, 10000, fs)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_nt_regcov_keep_uses_requested_pcs():
    # Dimensions: data N, regressor R
    N, R = 4, 5
    rng = np.random.default_rng(0)
    A = rng.standard_normal((R, R))
    cyy = A.T @ A + 1e-6 * np.eye(R)
    cxy = rng.standard_normal((N, R))  # as in usage, input is (N, R)

    zp = PyZaplinePlus(np.random.randn(100, N), 250, plotResults=False)
    r = zp.nt_regcov(cxy, cyy, keep=1, threshold=0)
    # Shape (R, N), rank at most 1 if only one PC kept
    assert r.shape == (R, N)
    assert np.linalg.matrix_rank(r) <= 1


def test_nt_dss0_pwr_shapes_and_finiteness():
    rng = np.random.default_rng(1)
    C = rng.standard_normal((6, 6))
    c0 = C.T @ C + 1e-6 * np.eye(6)
    # Biased covariance with a boosted first direction
    D = np.diag([2.0, 1.2, 1.1, 1.0, 0.9, 0.8])
    c1 = c0 @ D @ c0  # SPD-like transformation to change power distribution

    zp = PyZaplinePlus(np.random.randn(200, 6), 250, plotResults=False)
    todss, pwr0, pwr1 = zp.nt_dss0(c0, c1)
    assert todss.shape[0] == 6
    assert todss.shape[1] <= 6
    assert pwr0.shape == (todss.shape[1],)
    assert pwr1.shape == (todss.shape[1],)
    assert np.isfinite(pwr0).all() and np.isfinite(pwr1).all()
    # Scores finite
    scores = pwr1 / (pwr0 + 1e-30)
    assert np.isfinite(scores).all()

