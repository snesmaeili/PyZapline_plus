"""
Noise detection utilities for PyZaplinePlus.

This module contains functions for detecting line noise frequencies in EEG data.
"""

import numpy as np


def find_next_noisefreq(
    pxx,
    f,
    minfreq=0,
    threshdiff=4,
    winsizeHz=6,
    maxfreq=None,
    lower_threshdiff=1.76091259055681,
    verbose=False,
):
    """Locate the next candidate noise peak in a Welch spectrum.

    The detector mirrors the heuristics used in MATLAB zapline-plus: it scans the
    averaged power spectral density with a sliding window, compares the window centre
    to the outer thirds of the window, and flags peaks that exceed the coarse and
    fine thresholds expressed in dB.

    Parameters
    ----------
    pxx : ndarray
        Power spectral density array (frequency x channels).
    f : ndarray
        Frequency vector corresponding to ``pxx``.
    minfreq : float
        Minimum frequency to search from (default: 0).
    threshdiff : float
        Threshold difference for the initial detection phase (default: 4).
    winsizeHz : float
        Window size in Hz for the sliding analysis window (default: 6).
    maxfreq : float | None
        Maximum frequency to search to. If ``None`` it is set to 85 percent of the maximum
        frequency represented in ``f``.
    lower_threshdiff : float
        Threshold used to continue the detection once a peak has been found in the
        coarse stage (default: 1.76091259055681).
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    noisefreq : float | None
        Detected noise frequency, or ``None`` if no noise was found.
    thisfreqs : ndarray | None
        Frequency array of the last analysed window.
    thisdata : ndarray | None
        Power data (in dB) of the last analysed window.
    threshfound : float | None
        Coarse threshold used for the detected peak.
    """
    if maxfreq is None:
        maxfreq = float(np.max(f) * 0.85)

    winsize_bins = int(round(pxx.shape[0] / (float(np.max(f)) - float(np.min(f))) * winsizeHz))
    winsize_bins = max(winsize_bins, 3)
    half_win = winsize_bins // 2

    mean_psd = np.mean(pxx, axis=1)
    last_window_freqs = None
    last_window_data = None

    start_idx = np.searchsorted(f, minfreq, side='right')
    start_idx = max(start_idx, half_win)
    stop_idx = np.searchsorted(f, maxfreq, side='left')
    stop_idx = min(stop_idx, len(f) - half_win - 1)

    if start_idx >= stop_idx:
        if verbose:
            print("Search interval is empty; no noise detected.")
        return None, None, None, None

    peak_active = False
    peak_start = None
    peak_end = None
    threshfound = None
    verbose_marker = 0

    for window_start in range(start_idx - half_win, stop_idx - half_win + 1):
        window_stop = window_start + winsize_bins
        window_data = mean_psd[window_start:window_stop]
        window_freqs = f[window_start:window_stop]

        last_window_freqs = window_freqs
        last_window_data = window_data

        mid_offset = int(round(window_data.size / 2.0)) - 1
        centre_idx = window_start + mid_offset

        if verbose:
            current_hz = round(float(window_freqs[mid_offset]))
            if current_hz > verbose_marker:
                print(f"{current_hz},", end="")
                verbose_marker = current_hz

        third = max(int(round(window_data.size / 3.0)), 1)
        outer = np.concatenate((window_data[:third], window_data[-third:]))
        outer_mean = float(np.mean(outer))
        coarse_thresh = outer_mean + threshdiff
        fine_thresh = outer_mean + lower_threshdiff
        centre_val = float(window_data[mid_offset])

        if not peak_active:
            if centre_val > coarse_thresh:
                peak_active = True
                peak_start = centre_idx
                peak_end = centre_idx
                threshfound = coarse_thresh
        else:
            if centre_val > fine_thresh:
                peak_end = centre_idx
            else:
                slice_view = mean_psd[peak_start:peak_end + 1]
                peak_rel = int(np.argmax(slice_view))
                peak_idx = peak_start + peak_rel
                noisefreq = float(f[peak_idx])
                if verbose:
                    print(f"\nfound {noisefreq}Hz!")
                return noisefreq, last_window_freqs, last_window_data, threshfound

    if peak_active and peak_start is not None and peak_end is not None:
        slice_view = mean_psd[peak_start:peak_end + 1]
        peak_rel = int(np.argmax(slice_view))
        peak_idx = peak_start + peak_rel
        noisefreq = float(f[peak_idx])
        if verbose:
            print(f"\nfound {noisefreq}Hz at window boundary.")
        return noisefreq, last_window_freqs, last_window_data, threshfound

    if verbose:
        print("\nnone found.")
    return None, last_window_freqs, last_window_data, None
