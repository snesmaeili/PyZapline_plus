
import numpy as np
import pytest

from pyzaplineplus.noise_detection import find_next_noisefreq


def _make_test_psd(power_delta):
    f = np.linspace(0.0, 125.0, 501)
    pxx = np.zeros((f.size, 1))
    idx = int(np.argmin(np.abs(f - 50.0)))
    pxx[idx, 0] = power_delta
    return f, pxx, idx


def test_find_next_noisefreq_matches_matlab_threshold_and_window():
    f, pxx, idx = _make_test_psd(4.2)
    noisefreq, thisfreqs, thisdata, thresh = find_next_noisefreq(
        pxx, f, minfreq=45.0, maxfreq=55.0
    )
    assert noisefreq is not None
    assert noisefreq == pytest.approx(f[idx])

    winsize_expected = int(round(pxx.shape[0] / (f.max() - f.min()) * 6.0))
    winsize_expected = max(winsize_expected, 3)
    assert len(thisfreqs) == winsize_expected

    # Background is 0 dB, so threshold should equal the 4 dB coarse difference
    assert thresh == pytest.approx(4.0, abs=1e-8)


def test_find_next_noisefreq_respects_threshold():
    f, pxx, _ = _make_test_psd(3.9)
    noisefreq, _, _, _ = find_next_noisefreq(pxx, f, minfreq=45.0, maxfreq=55.0)
    assert noisefreq is None

def test_find_next_noisefreq_respects_minfreq_window_edge():
    f = np.linspace(40.0, 60.0, 81)
    pxx = np.zeros((f.size, 1))
    target_idx = np.argmin(np.abs(f - 45.0))
    pxx[target_idx, 0] = 6.5

    noisefreq, _, _, _ = find_next_noisefreq(
        pxx,
        f,
        minfreq=44.8,
        maxfreq=55.0,
        winsizeHz=6.0,
    )
    assert noisefreq == pytest.approx(f[target_idx])


def test_find_next_noisefreq_handles_sparse_frequency_grid():
    f = np.linspace(45.0, 55.0, 6)
    pxx = np.zeros((f.size, 1))
    pxx[2, 0] = 7.0

    noisefreq, window_freqs, window_data, thresh = find_next_noisefreq(
        pxx,
        f,
        minfreq=44.0,
        maxfreq=56.0,
        winsizeHz=6.0,
    )

    assert noisefreq == pytest.approx(f[2])
    assert window_freqs is not None and window_freqs.size >= 3
    assert window_data is not None and window_data.size == window_freqs.size
    assert thresh > 0

def test_find_next_noisefreq_tracks_drifting_peak():
    f = np.linspace(40.0, 70.0, 601)
    drift_freq = 50.4
    pxx = np.zeros((f.size, 1))
    idx = np.argmin(np.abs(f - drift_freq))
    pxx[idx, 0] = 6.0

    noisefreq, _, _, _ = find_next_noisefreq(
        pxx,
        f,
        minfreq=45.0,
        maxfreq=60.0,
        winsizeHz=6.0,
    )

    bin_width = f[1] - f[0]
    assert noisefreq == pytest.approx(f[idx], abs=bin_width)
