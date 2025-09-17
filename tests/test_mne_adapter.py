import sys
import numpy as np
import pytest
from pyzaplineplus import apply_zapline_to_raw, zapline_plus_epochs, zapline_plus_raw


def _psd_50hz_bin_db(x, fs, nperseg):
    from pyzaplineplus.core import _welch_hamming_periodic

    f, pxx = _welch_hamming_periodic(x, fs=fs, nperseg=nperseg, axis=-1)
    pxx_db = 10 * np.log10(np.mean(pxx, axis=0)) if pxx.ndim == 2 else 10 * np.log10(pxx)
    # Find index nearest 50 Hz
    idx = int(np.argmin(np.abs(f - 50.0)))
    return float(pxx_db[idx])


@pytest.mark.integration
def test_mne_raw_eeg_cleaning_reduces_50hz():
    mne = pytest.importorskip("mne")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(42)
    fs = 500.0
    dur = 20.0
    n_ch = 8
    t = np.arange(0, dur, 1.0 / fs)
    sine = np.sin(2 * np.pi * 50.0 * t)
    data = 0.05 * rng.standard_normal((n_ch, t.size)) + 0.5 * sine[np.newaxis, :]

    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, copy="auto")

    nperseg = 512
    before_db = _psd_50hz_bin_db(data, fs, nperseg)

    raw_out, config, analytics, figs = zapline_plus_raw(
        raw,
        line_freqs=[50.0],
        picks=None,
        copy=True,
        plotResults=False,
        adaptiveNremove=True,
    )

    after_db = _psd_50hz_bin_db(raw_out.get_data(), fs, nperseg)

    # Expect at least 15 dB reduction
    assert before_db - after_db >= 15.0

    # Metadata unchanged
    assert len(raw_out.ch_names) == len(raw.ch_names)
    assert raw_out.info["sfreq"] == raw.info["sfreq"]
    assert len(raw_out.annotations) == len(raw.annotations)


@pytest.mark.integration
def test_mne_picks_respected():
    mne = pytest.importorskip("mne")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(0)
    fs = 500.0
    dur = 10.0
    n_ch = 6
    t = np.arange(0, dur, 1.0 / fs)
    data = 0.05 * rng.standard_normal((n_ch, t.size))
    # Inject 50 Hz only in last two channels
    sine = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    data[-2:, :] += sine[np.newaxis, :]

    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, copy="auto")

    last_two = raw.ch_names[-2:]

    nperseg = 512
    before_per_ch = []
    for ch in range(n_ch):
        before_per_ch.append(_psd_50hz_bin_db(data[ch, :], fs, nperseg))

    raw_out, *_ = zapline_plus_raw(
        raw,
        line_freqs=[50.0],
        picks=last_two,
        copy=True,
        plotResults=False,
        adaptiveNremove=True,
    )

    after_per_ch = []
    out_data = raw_out.get_data()
    for ch in range(n_ch):
        after_per_ch.append(_psd_50hz_bin_db(out_data[ch, :], fs, nperseg))

    # Picked channels reduced
    for ch in [-2, -1]:
        assert before_per_ch[ch] - after_per_ch[ch] >= 12.0
    # Unpicked channels unchanged within ±1 dB
    for ch in range(n_ch - 2):
        assert abs(before_per_ch[ch] - after_per_ch[ch]) <= 1.0


def test_mne_copy_false_inplace():
    mne = pytest.importorskip("mne")
    import matplotlib

    matplotlib.use("Agg")

    fs = 500.0
    dur = 5.0
    n_ch = 4
    t = np.arange(0, dur, 1.0 / fs)
    sine = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    data = 0.05 * np.random.randn(n_ch, t.size) + sine[np.newaxis, :]
    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, copy="auto")

    raw_id = id(raw)
    raw_out, *_ = zapline_plus_raw(raw, line_freqs=[50.0], picks=None, copy=False, plotResults=False)

    assert id(raw_out) == raw_id
    # Data changed
    assert np.linalg.norm(raw_out.get_data() - data) > 1e-3


def test_mne_dtype_roundtrip():
    mne = pytest.importorskip("mne")
    fs = 500.0
    dur = 5.0
    n_ch = 3
    t = np.arange(0, dur, 1.0 / fs)
    sine = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    data32 = (0.05 * np.random.randn(n_ch, t.size) + sine[np.newaxis, :]).astype(np.float32)
    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data32.copy(), info, copy="auto")

    dtype_before = raw._data.dtype
    raw_out, *_ = zapline_plus_raw(raw, line_freqs=[50.0], picks=None, copy=False, plotResults=False)
    assert raw_out._data.dtype == dtype_before


@pytest.mark.integration
def test_mne_epochs_cleaning_reduces_50hz():
    mne = pytest.importorskip("mne")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(1)
    fs = 500.0
    n_epochs = 6
    dur = 2.0
    n_ch = 5
    n_times = int(dur * fs)
    t = np.arange(n_times) / fs
    sine = np.sin(2 * np.pi * 50.0 * t)
    data = 0.05 * rng.standard_normal((n_epochs, n_ch, n_times))
    data += 0.4 * sine[np.newaxis, np.newaxis, :]

    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data.copy(), info, verbose="ERROR")

    nperseg = 256
    before_db = _psd_50hz_bin_db(data.transpose(1, 0, 2).reshape(n_ch, -1), fs, nperseg)

    epochs_out, config, analytics, figs = zapline_plus_epochs(
        epochs,
        line_freqs=[50.0],
        picks=None,
        copy=True,
        plotResults=False,
        adaptiveNremove=True,
    )

    after_data = epochs_out.get_data().transpose(1, 0, 2).reshape(n_ch, -1)
    after_db = _psd_50hz_bin_db(after_data, fs, nperseg)
    assert before_db - after_db >= 12.0

    assert epochs_out.info["sfreq"] == epochs.info["sfreq"]
    assert epochs_out.events.shape == epochs.events.shape


@pytest.mark.integration
def test_mne_epochs_picks_respected():
    mne = pytest.importorskip("mne")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(3)
    fs = 500.0
    n_epochs = 4
    dur = 1.5
    n_ch = 6
    n_times = int(dur * fs)
    t = np.arange(n_times) / fs
    sine = 0.4 * np.sin(2 * np.pi * 50.0 * t)
    data = 0.05 * rng.standard_normal((n_epochs, n_ch, n_times))
    data[:, -2:, :] += sine[np.newaxis, np.newaxis, :]

    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data.copy(), info, verbose="ERROR")

    last_two = epochs.ch_names[-2:]
    nperseg = 256

    before_per_ch = []
    for ch in range(n_ch):
        before_per_ch.append(_psd_50hz_bin_db(data[:, ch, :].reshape(-1), fs, nperseg))

    epochs_out, *_ = zapline_plus_epochs(
        epochs,
        line_freqs=[50.0],
        picks=last_two,
        copy=True,
        plotResults=False,
        adaptiveNremove=True,
    )

    after = epochs_out.get_data()
    after_per_ch = []
    for ch in range(n_ch):
        after_per_ch.append(_psd_50hz_bin_db(after[:, ch, :].reshape(-1), fs, nperseg))

    for ch in [-2, -1]:
        assert before_per_ch[ch] - after_per_ch[ch] >= 10.0
    for ch in range(n_ch - 2):
        assert abs(before_per_ch[ch] - after_per_ch[ch]) <= 1.0


@pytest.mark.integration
def test_mne_epochs_copy_false_inplace():
    mne = pytest.importorskip("mne")
    import matplotlib

    matplotlib.use("Agg")

    fs = 500.0
    n_epochs = 3
    dur = 1.0
    n_ch = 4
    n_times = int(dur * fs)
    t = np.arange(n_times) / fs
    sine = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    data = 0.05 * np.random.randn(n_epochs, n_ch, n_times) + sine[np.newaxis, np.newaxis, :]

    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data.copy(), info, verbose="ERROR")

    epochs_id = id(epochs)
    epochs_out, *_ = zapline_plus_epochs(epochs, line_freqs=[50.0], picks=None, copy=False, plotResults=False)

    assert id(epochs_out) == epochs_id
    assert np.linalg.norm(epochs_out.get_data() - data) > 1e-3


@pytest.mark.integration
def test_mne_epochs_dtype_roundtrip():
    mne = pytest.importorskip("mne")

    fs = 500.0
    n_epochs = 2
    dur = 1.0
    n_ch = 3
    n_times = int(dur * fs)
    t = np.arange(n_times) / fs
    sine = 0.4 * np.sin(2 * np.pi * 50.0 * t)
    data = (0.05 * np.random.randn(n_epochs, n_ch, n_times) + sine[np.newaxis, np.newaxis, :]).astype(np.float32)

    info = mne.create_info(ch_names=[f"EEG{i:02d}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data.copy(), info, verbose="ERROR")

    dtype_before = epochs._data.dtype
    epochs_out, *_ = zapline_plus_epochs(epochs, line_freqs=[50.0], picks=None, copy=False, plotResults=False)
    assert epochs_out._data.dtype == dtype_before


def test_apply_alias_matches_new_wrapper():
    assert apply_zapline_to_raw is zapline_plus_raw

def test_missing_mne_guard(monkeypatch):
    import importlib

    # Simulate missing mne at import time inside the adapter function
    monkeypatch.setitem(sys.modules, "mne", None)
    from pyzaplineplus import _mne as adapter

    class Dummy:
        preload = True

    with pytest.raises(ImportError) as ei:
        adapter.zapline_plus_raw(Dummy())
    msg = str(ei.value)
    assert "pip install" in msg and "mne" in msg

