import numpy as np

def find_next_noisefreq(pxx, f, minfreq=0, threshdiff=5, winsizeHz=3, maxfreq=None, lower_threshdiff=1.76091259055681, verbose=False):
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
            max_value = np.max(meandata[i_startdetected:i_enddetected + 1])
            max_indices = np.where(meandata[i_startdetected:i_enddetected + 1] == max_value)[0]
            noisefreq = f[max_indices[0] + i_startdetected]
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
