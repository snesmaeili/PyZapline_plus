import numpy as np

def find_next_noisefreq(pxx, f, minfreq=0, threshdiff=5, winsizeHz=3, maxfreq=None, lower_threshdiff=1.76091259055681, verbose=False):
    """
    Search for the next noise frequency based on the spectrum starting from a minimum frequency.
    
    Args:
    pxx (np.array): Power spectral density (in log space)
    f (np.array): Frequency array
    minfreq (float): Minimum frequency to consider
    threshdiff (float): Threshold difference for peak detection
    winsizeHz (float): Window size in Hz
    maxfreq (float): Maximum frequency to consider
    lower_threshdiff (float): Lower threshold difference
    verbose (bool): If True, print debug information
    
    Returns:
    tuple: (noisefreq, thisfreqs, thisdata, threshfound)
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

    i_start = max(np.argmax(f > minfreq) + 1, round(winsize / 2))
    i_end = min(np.argmax(f >= maxfreq), len(f) - round(winsize / 2))

    lastfreq = 0
    for i in range(i_start - round(winsize / 2), i_end - round(winsize / 2) + 1):
        thisdata = meandata[i:i+winsize]
        thisfreqs = f[i:i+winsize]

        thisfreq = round(thisfreqs[round(len(thisfreqs) / 2)])
        if verbose and thisfreq > lastfreq:
            print(f"{thisfreq},", end="")
            lastfreq = thisfreq

        third = round(len(thisdata) / 3)
        center_thisdata = np.mean(np.concatenate([thisdata[:third], thisdata[2*third:]]))
        thresh = center_thisdata + threshdiff

        if not detected:
            detectednew = thisdata[round(len(thisdata) / 2)] > thresh
            if detectednew:
                i_startdetected = round(i + (winsize - 1) / 2)
                threshfound = thresh
        else:
            detectednew = thisdata[round(len(thisdata) / 2)] > center_thisdata + lower_threshdiff
            i_enddetected = round(i + (winsize - 1) / 2)

        if not detectionstart and detected and not detectednew:
            detectionstart = True
        elif detectionstart and detected and not detectednew:
            noisefreq = f[np.argmax(meandata[i_startdetected:i_enddetected+1]) + i_startdetected]
            if verbose:
                print(f"\nfound {noisefreq}Hz!")
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(thisfreqs, thisdata)
                plt.axhline(y=thresh, color='r', linestyle='-')
                plt.axhline(y=center_thisdata, color='k', linestyle='-')
                plt.title(str(noisefreq))
                plt.show()
            return noisefreq, thisfreqs, thisdata, threshfound

        detected = detectednew

    if verbose:
        print("\nnone found.")
    return noisefreq, thisfreqs, thisdata, threshfound