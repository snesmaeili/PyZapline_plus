"""
MNE EEGBCI demo for PyZaplinePlus.

This example loads a 60-second EEG segment from the MNE EEGBCI dataset,
attempts to detect and remove line noise with PyZaplinePlus, and generates
the result visualization similar to MATLAB's zapline-plus figures.

If no clear line noise is detected (common on some short/clean segments),
the script prints a summary and still generates a 'No noise found' figure.

Usage:
  python3 examples/mne_eegbci_demo.py
"""

import os
import sys
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from scipy import signal

# Ensure repository root is on sys.path for local imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pyzaplineplus import zapline_plus


def main():
    mne.set_log_level('WARNING')

    # Load subject 1, run 3 (eyes open), download if needed
    subject = 1
    runs = [3]
    paths = eegbci.load_data(subject, runs, update_path=True)

    raw = read_raw_edf(paths[0], preload=True, verbose=False)
    raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False)
    raw.crop(tmin=0, tmax=60.0)

    # Optional: resample to <= 500 Hz as recommended
    if raw.info['sfreq'] > 500:
        raw.resample(500)

    fs = raw.info['sfreq']
    data = raw.get_data()  # shape: (n_channels, n_times)
    print(f"Data shape: {data.shape}, fs={fs}")

    # Run zapline_plus with visualization enabled
    # Use a slightly more permissive threshold (3 dB) for automatic detection
    clean_data, config, analytics, plots = zapline_plus(
        data, fs,
        noisefreqs='line',
        coarseFreqDetectPowerDiff=3,
        plotResults=True,
        chunkLength=0
    )

    # Report detected frequencies
    detected = config.get('noisefreqs', [])
    print('Detected noise frequencies:', detected)

    # Evaluate reduction at first detected frequency if any
    if len(detected) > 0:
        target = float(detected[0])
        f, pxx_o = signal.welch(data.T, fs=fs, axis=0)
        f2, pxx_c = signal.welch(clean_data.T, fs=fs, axis=0)
        fi = np.argmin(np.abs(f - target))
        before = float(np.mean(pxx_o[fi, :]))
        after = float(np.mean(pxx_c[fi, :]))
        print(f"Power at {target:.2f} Hz – before: {before:.4g}, after: {after:.4g}, ratio={after/before:.3f}")
    else:
        print('No clear line noise detected in this short segment (figure still generated).')
        # Optional demo: inject synthetic 50 Hz line noise to showcase full visualization
        print('Injecting 50 Hz synthetic line noise for demonstration...')
        n_times = data.shape[1]
        t = np.arange(n_times) / fs
        noise = 5.0 * np.sin(2 * np.pi * 50 * t)
        noisy = data + noise[np.newaxis, :]
        clean_demo, config_demo, _, _ = zapline_plus(
            noisy, fs,
            noisefreqs='line',
            coarseFreqDetectPowerDiff=3,
            plotResults=True,
            chunkLength=0
        )
        f, pxx_o = signal.welch(noisy.T, fs=fs, axis=0)
        f2, pxx_c = signal.welch(clean_demo.T, fs=fs, axis=0)
        fi = np.argmin(np.abs(f - 50.0))
        before = float(np.mean(pxx_o[fi, :]))
        after = float(np.mean(pxx_c[fi, :]))
        print(f"[Demo] Power at 50.00 Hz – before: {before:.4g}, after: {after:.4g}, ratio={after/before:.3f}")

    out_png = os.path.join('figures', 'zapline_results.png')
    if os.path.isfile(out_png):
        print(f'Visualization saved to: {out_png}')
    else:
        print('Visualization not found on disk (unexpected).')


if __name__ == '__main__':
    main()
