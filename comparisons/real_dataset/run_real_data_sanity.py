"""Run MATLAB vs Python Zapline-plus parity on a real dataset with line noise.

This script download a Brainstorm tutorial MEG dataset (if not already
present), extracts a 60-second segment with strong 60 Hz line noise, runs the
PyZaplinePlus implementation, calls the MATLAB reference implementation via the
MATLAB Engine for Python, and saves PSD comparison plots plus basic metrics.

Execute with the ``matpy`` environment so both MNE and the MATLAB engine are
available::

    /mnt/c/Users/s/anaconda3/envs/matpy/python.exe comparisons/real_dataset/run_real_data_sanity.py

Outputs are written to ``comparisons/real_dataset/results``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

# Use non-interactive backend for predictable file output
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_data() -> Tuple[np.ndarray, float, list[str]]:
    """Load the Brainstorm raw dataset and return (samples, channels) data."""

    import mne

    repo_root = _repo_root()
    data_root = repo_root / "comparisons" / "real_dataset" / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Download (or locate) the Brainstorm raw dataset; accept license
    bst_dir = Path(
        mne.datasets.brainstorm.bst_raw.data_path(
            path=str(data_root), accept=True, verbose=False
        )
    )
    ds_path = next(
        bst_dir.rglob("subj001_somatosensory_20111109_01_AUX-f.ds")
    )

    raw = mne.io.read_raw_ctf(
        ds_path,
        system_clock="ignore",
        preload=True,
        verbose="ERROR",
    )

    raw.pick_types(meg=True, ref_meg=False, stim=False, misc=False)

    # Focus on a 60-second window away from onset transients.
    raw.crop(tmin=60.0, tmax=120.0)
    raw.resample(300.0, npad="auto")

    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    data = raw.get_data().T  # -> samples x channels

    return data, sfreq, channel_names


def _compute_psd_db(data: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """Average Welch PSD (dB) across channels using MATLAB-matching settings."""

    if data.ndim != 2:
        raise ValueError("PSD input must be 2D (samples x channels)")

    nperseg = min(data.shape[0], max(int(sfreq * 5.0), 256))
    win = signal.windows.hamming(nperseg, sym=False)
    f, pxx = signal.welch(
        data,
        fs=sfreq,
        window=win,
        noverlap=nperseg // 2,
        nperseg=nperseg,
        detrend=False,
        return_onesided=True,
        axis=0,
    )
    pxx_db = 10.0 * np.log10(np.maximum(pxx, np.finfo(float).tiny))
    return f, pxx_db.mean(axis=1)


def _run_python(data: np.ndarray, sfreq: float) -> np.ndarray:
    from pyzaplineplus import PyZaplinePlus

    zap = PyZaplinePlus(
        data,
        sfreq,
        noisefreqs="line",
        plotResults=False,
        adaptiveNremove=True,
        searchIndividualNoise=True,
    )
    clean_data, _, _, _ = zap.run()
    return clean_data


def _run_matlab(data: np.ndarray, sfreq: float) -> np.ndarray:
    import matlab.engine

    repo_root = _repo_root()
    matlab_path = repo_root / "zapline-plus"
    eng = matlab.engine.start_matlab()
    try:
        eng.addpath(matlab_path.as_posix(), nargout=0)
        matlab_data = matlab.double(data.tolist())
        clean_data, _, _, _ = eng.clean_data_with_zapline_plus(
            matlab_data,
            float(sfreq),
            "noisefreqs",
            "line",
            "plotResults",
            False,
            "saveSpectra",
            False,
            nargout=4,
        )
    finally:
        eng.quit()

    clean_array = np.array(clean_data, dtype=float)
    return clean_array


def _save_outputs(
    out_dir: Path,
    raw_data: np.ndarray,
    clean_py: np.ndarray,
    clean_mat: np.ndarray,
    sfreq: float,
    channel_names: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "raw_clip.npy", raw_data)
    np.save(out_dir / "clean_python.npy", clean_py)
    np.save(out_dir / "clean_matlab.npy", clean_mat)

    f_raw, psd_raw = _compute_psd_db(raw_data, sfreq)
    _, psd_python = _compute_psd_db(clean_py, sfreq)
    _, psd_matlab = _compute_psd_db(clean_mat, sfreq)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f_raw, psd_raw, label="Raw", color="tab:gray")
    ax.plot(f_raw, psd_python, label="PyZaplinePlus", color="tab:blue")
    ax.plot(f_raw, psd_matlab, label="MATLAB zapline-plus", color="tab:orange")
    ax.axvline(60.0, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "psd_comparison.png", dpi=150)
    plt.close(fig)

    # Metrics for sanity checking
    diff = clean_py - clean_mat
    metrics = {
        "sfreq": sfreq,
        "n_samples": int(raw_data.shape[0]),
        "n_channels": int(raw_data.shape[1]),
        "channel_names": channel_names,
        "relative_l2_clean_diff": float(
            np.linalg.norm(diff) / np.linalg.norm(raw_data)
        ),
        "raw_line_bin_db": float(
            psd_raw[np.argmin(np.abs(f_raw - 60.0))]
        ),
        "python_line_bin_db": float(
            psd_python[np.argmin(np.abs(f_raw - 60.0))]
        ),
        "matlab_line_bin_db": float(
            psd_matlab[np.argmin(np.abs(f_raw - 60.0))]
        ),
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    out_dir = repo_root / "comparisons" / "real_dataset" / "results"

    raw_data, sfreq, ch_names = _prepare_data()
    clean_python = _run_python(raw_data, sfreq)
    clean_matlab = _run_matlab(raw_data, sfreq)
    _save_outputs(out_dir, raw_data, clean_python, clean_matlab, sfreq, ch_names)


if __name__ == "__main__":
    main()
