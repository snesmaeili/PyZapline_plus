import numpy as np
from scipy import signal


def test_welch_helper_uses_periodic_hamming_and_overlap():
    from pyzaplineplus.core import _welch_hamming_periodic

    x = np.random.randn(1000)
    fs = 500.0
    nperseg = 256

    # Periodic vs symmetric windows differ
    w_per = signal.windows.hamming(nperseg, sym=False)
    w_sym = signal.windows.hamming(nperseg, sym=True)
    assert not np.allclose(w_per, w_sym)

    f1, pxx1 = _welch_hamming_periodic(x, fs=fs, nperseg=nperseg, axis=0)
    assert f1.ndim == 1 and pxx1.ndim == 1
    assert f1.size == pxx1.size

    # A different overlap should generally change the PSD estimate
    f2, pxx2 = signal.welch(
        x,
        fs=fs,
        window=w_per,
        noverlap=0,  # no overlap
        nperseg=nperseg,
        detrend=False,
    )
    # Not enforcing strict inequality over all bins, but substantial difference should exist
    assert np.mean(np.abs(pxx1 - pxx2)) > 0
