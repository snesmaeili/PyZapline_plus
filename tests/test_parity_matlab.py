"""
Parity and guard tests for MATLAB zapline-plus behavior.

Covers PSD windowing, small-chunk guards, Nyquist guards, and nt_* fixes.
"""

import numpy as np
import pytest
from scipy import signal
from scipy.signal import windows

from pyzaplineplus import PyZaplinePlus


@pytest.mark.parametrize("fs", [125, 250, 500])
def test_psd_parity_hamming_peak_bin(fs):
    duration = 20.0
    rng = np.random.default_rng(0)
    t = np.arange(0, duration, 1 / fs)
    x = 0.2 * rng.standard_normal((t.size, 4))
    x += 1.5 * np.sin(2 * np.pi * 50 * t)[:, None]

    win_sec = 1.0
    nperseg = int(win_sec * fs)
    zp = PyZaplinePlus(x, fs, plotResults=False, winSizeCompleteSpectrum=win_sec)
    pxx_log, f = zp.compute_spectrum(x)

    ref_win = windows.hamming(nperseg, sym=False)
    f_ref, pxx_ref = signal.welch(
        x, fs=fs, window=ref_win, nperseg=nperseg, noverlap=nperseg // 2, axis=0
    )

    mean_pxx = np.mean(10 * np.log10(pxx_ref), axis=1)
    mean_pxx_zp = np.mean(pxx_log, axis=1)
    band = (f_ref >= 40) & (f_ref <= 60)
    idx_ref = np.argmax(mean_pxx[band])
    idx_zp = np.argmax(mean_pxx_zp[band])
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



def test_iterative_outlier_removal_counts_outliers():
    rng = np.random.default_rng(0)
    zap = PyZaplinePlus(rng.standard_normal((200, 2)), 250, plotResults=False)
    scores = np.concatenate([np.ones(20), np.array([5.0, 6.0])])
    n_remove, threshold = zap.iterative_outlier_removal(scores.copy(), sd_level=1.0)
    assert n_remove == 2
    assert threshold == pytest.approx(1.0)

def test_adaptive_chunk_detection_detects_covariance_shift():
    fs = 100
    segment_len = 0.5
    line = 10.0
    seg_samples = int(segment_len * fs)
    t = np.arange(seg_samples) / fs
    sine = np.sin(2 * np.pi * line * t)
    segment_a = np.column_stack([sine, 0.1 * sine])
    segment_b = np.column_stack([sine, 0.1 * sine])
    segment_c = np.column_stack([0.05 * sine, 2.0 * sine])
    segment_d = np.column_stack([0.05 * sine, 2.0 * sine])
    data = np.vstack([segment_a, segment_b, segment_c, segment_d])
    data += 1e-3 * np.random.default_rng(1).standard_normal(data.shape)

    zap = PyZaplinePlus(
        data,
        fs,
        plotResults=False,
        segmentLength=segment_len,
        minChunkLength=segment_len,
        chunkLength=0,
        prominenceQuantile=0.5,
    )
    chunks = zap.adaptive_chunk_detection(line)
    assert chunks == [0, seg_samples * 2, data.shape[0]]

def test_detect_noise_frequencies_identifies_dual_line_peaks():
    fs = 500
    duration = 30.0
    rng = np.random.default_rng(1)
    t = np.arange(0, duration, 1 / fs)
    x = 0.1 * rng.standard_normal((t.size, 2))
    x += 1.2 * np.sin(2 * np.pi * 50 * t)[:, None]
    x += 0.8 * np.sin(2 * np.pi * 60 * t)[:, None]

    zap = PyZaplinePlus(
        x,
        fs,
        plotResults=False,
        minfreq=40,
        maxfreq=70,
    )
    zap.finalize_inputs()
    detected = np.array(zap.config['noisefreqs'], dtype=float)

    assert detected.size >= 2
    freq_resolution = zap.f[1] - zap.f[0] if zap.f.size > 1 else 0.0

    def _contains(target: float) -> bool:
        if freq_resolution == 0.0:
            return np.any(np.isclose(detected, target))
        return np.any(np.abs(detected - target) <= freq_resolution)

    assert _contains(50.0)
    assert _contains(60.0)



def test_adaptive_cleaning_handles_two_bin_window():
    fs = 250
    duration = 0.2
    rng = np.random.default_rng(2)
    t = np.arange(0, duration, 1 / fs)
    data = 0.05 * rng.standard_normal((t.size, 2))
    data += np.sin(2 * np.pi * 50 * t)[:, None]

    zap = PyZaplinePlus(data, fs, plotResults=False)
    zap.finalize_inputs()
    zapline_config = zap.config.copy()
    clean_data = zap.data.copy()

    cleaning_done, updated_config, cleaning_flag = zap.adaptive_cleaning(
        clean_data,
        zap.data,
        50.0,
        zapline_config,
        cleaning_too_strong_once=False,
    )

    assert isinstance(cleaning_done, bool)
    assert cleaning_flag in (True, False)
    assert np.isfinite(updated_config['remaining_noise_thresh_upper'])
    assert np.isfinite(updated_config['remaining_noise_thresh_lower'])
    assert 'thisFreqidxUppercheck' in updated_config
    assert updated_config['thisFreqidxUppercheck'].dtype == bool
