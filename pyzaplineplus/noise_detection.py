"""
Noise detection utilities for PyZaplinePlus.

This module contains functions for detecting line noise frequencies in EEG data.
"""

import numpy as np


def find_next_noisefreq(pxx, f, minfreq=0, threshdiff=5, winsizeHz=3, maxfreq=None,
                       lower_threshdiff=1.76091259055681, verbose=False):
    """
    Find the next noise frequency in the power spectrum.

    This function searches for noise frequencies by analyzing the power spectral density
    and identifying peaks that exceed a threshold relative to surrounding frequencies.

    Parameters
    ----------
    pxx : ndarray
        Power spectral density array (frequency x channels).
    f : ndarray
        Frequency vector corresponding to pxx.
    minfreq : float, optional
        Minimum frequency to search from (default: 0).
    threshdiff : float, optional
        Threshold difference for noise detection (default: 5).
    winsizeHz : float, optional
        Window size in Hz for analysis (default: 3).
    maxfreq : float, optional
        Maximum frequency to search to. If None, uses 85% of max frequency.
    lower_threshdiff : float, optional
        Lower threshold difference for continued detection (default: 1.76091259055681).
    verbose : bool, optional
        Whether to print verbose output and show plots (default: False).

    Returns
    -------
    noisefreq : float or None
        Detected noise frequency, or None if no noise found.
    thisfreqs : ndarray or None
        Frequency array of the analyzed window.
    thisdata : ndarray or None
        Power data of the analyzed window.
    threshfound : float or None
        The threshold that was used for detection.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> # Create sample data with line noise
    >>> fs = 1000
    >>> t = np.arange(0, 10, 1/fs)
    >>> data = np.random.randn(len(t), 32) + 0.5 * np.sin(2*np.pi*50*t)[:, np.newaxis]
    >>> f, pxx = signal.welch(data, fs=fs, axis=0)
    >>> noisefreq, _, _, _ = find_next_noisefreq(pxx, f, minfreq=45, maxfreq=55)
    >>> print(f"Detected noise at {noisefreq:.2f} Hz")
    """
    if maxfreq is None:
        maxfreq = max(f) * 0.85

    if verbose:
        print(f"Searching for first noise freq between {minfreq}Hz and {maxfreq}Hz...")

    noisefreq = None
    threshfound = None
    thisfreqs = None
    thisdata = None
    winsize = round(pxx.shape[0] / (max(f) - min(f)) * winsizeHz)
    meandata = np.mean(pxx, axis=1)

    detectionstart = False
    detected = True
    i_startdetected = 0
    i_enddetected = 0

    # Compute i_start
    indices = np.where(f > minfreq)[0]
    if indices.size == 0:
        i_start = round(winsize / 2)
    else:
        i_start = max(indices[0] + 1, round(winsize / 2))

    # Compute i_end
    indices = np.where(f < maxfreq)[0]
    if indices.size == 0:
        i_end = len(f) - round(winsize / 2)
    else:
        i_end = min(indices[-1], len(f) - round(winsize / 2))

    lastfreq = 0
    for i in range(int(i_start - round(winsize / 2)), int(i_end - round(winsize / 2) + 1)):
        thisdata = meandata[i:i + winsize]
        thisfreqs = f[i:i + winsize]

        # Correct index for zero-based indexing
        middle_index = int(round(len(thisdata) / 2)) - 1
        thisfreq = round(thisfreqs[middle_index])

        if verbose and thisfreq > lastfreq:
            print(f"{thisfreq},", end="")
            lastfreq = thisfreq

        third = round(len(thisdata) / 3)
        center_thisdata = np.mean(np.concatenate([thisdata[:third], thisdata[2 * third:]]))
        thresh = center_thisdata + threshdiff

        if not detected:
            detectednew = thisdata[middle_index] > thresh
            if detectednew:
                i_startdetected = round(i + (winsize - 1) / 2)
                threshfound = thresh
        else:
            detectednew = thisdata[middle_index] > center_thisdata + lower_threshdiff
            i_enddetected = round(i + (winsize - 1) / 2)

        if not detectionstart and detected and not detectednew:
            detectionstart = True
        elif detectionstart and detected and not detectednew:
            # Handle multiple maxima
            max_value: float = float(np.max(meandata[i_startdetected:i_enddetected + 1]))
            max_indices = np.where(meandata[i_startdetected:i_enddetected + 1] == max_value)[0]
            noisefreq = f[max_indices[0] + i_startdetected]
            if verbose:
                print(f"\nfound {noisefreq}Hz!")
                try:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(thisfreqs, thisdata)
                    plt.axhline(y=float(thresh), color='r', linestyle='-')
                    plt.axhline(y=float(center_thisdata), color='k', linestyle='-')
                    plt.title(str(noisefreq))
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Power (dB)')
                    plt.show()
                except ImportError:
                    print("Matplotlib not available for plotting")
            return noisefreq, thisfreqs, thisdata, threshfound

        detected = detectednew

    if verbose:
        print("\nnone found.")
    return noisefreq, thisfreqs, thisdata, threshfound
