import numpy as np
import pytest


def test_short_chunk_guard_skips_removal():
    from pyzaplineplus.core import PyZaplinePlus

    fs = 500.0
    # Very short chunk relative to winSizeCompleteSpectrum (1 s => 500 samples; require >= 1000)
    data = 0.01 * np.random.randn(300, 4)
    z = PyZaplinePlus(
        data,
        fs,
        noisefreqs=[50.0],
        winSizeCompleteSpectrum=1,  # seconds
        plotResults=False,
    )
    # Call directly on a chunk that is too short
    with pytest.warns(RuntimeWarning, match="chunk too short"):
        clean_chunk, nremove, scores = z.apply_zapline_to_chunk(data, 50.0)
    assert nremove == 0
    assert clean_chunk.shape == data.shape
    assert scores.size == 0


def test_nt_regcov_keep_caps_rank():
    from pyzaplineplus.core import PyZaplinePlus

    # Dummy instance just to access method
    dummy = PyZaplinePlus(np.random.randn(100, 2), 200, plotResults=False, noisefreqs=[50])

    k = 4
    n = 3
    rng = np.random.default_rng(0)
    A = rng.standard_normal((k, k))
    cyy = A @ A.T + np.eye(k) * 1e-3  # SPD
    cxy = rng.standard_normal((n, k))

    r = dummy.nt_regcov(cxy, cyy, keep=1)
    # Rank-1 projection behavior
    assert np.linalg.matrix_rank(r) == 1

