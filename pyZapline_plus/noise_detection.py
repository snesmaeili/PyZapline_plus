import numpy as np
from scipy import signal

def detect_noise_frequencies(data, sampling_rate, config):
    """
    Detect noise frequencies in the data using Welch's method.
    
    Args:
    data (np.array): Input data, shape (channels, samples)
    sampling_rate (float): Sampling rate of the data
    config (dict): Configuration parameters
    
    Returns:
    list: Detected noise frequencies
    """
    minfreq = config['minfreq']
    maxfreq = config['maxfreq']
    detectionWinsize = config['detectionWinsize']
    coarseFreqDetectPowerDiff = config['coarseFreqDetectPowerDiff']
    coarseFreqDetectLowerPowerDiff = config['coarseFreqDetectLowerPowerDiff']

    # Compute PSD using Welch's method
    f, psd = signal.welch(data, fs=sampling_rate, window='hann', nperseg=sampling_rate)

    # Log-transform PSD and compute mean across channels
    log_psd = 10 * np.log10(psd)
    mean_log_psd = np.mean(log_psd, axis=0)

    # Find frequencies within the specified range
    freq_mask = (f >= minfreq) & (f <= maxfreq)
    freqs = f[freq_mask]
    mean_log_psd = mean_log_psd[freq_mask]

    noise_freqs = []
    i = 0
    while i < len(freqs):
        # Define window for local PSD analysis
        window_start = max(0, i - detectionWinsize // 2)
        window_end = min(len(freqs), i + detectionWinsize // 2)
        window = mean_log_psd[window_start:window_end]

        # Compute center power (mean of left and right thirds)
        third = len(window) // 3
        center_power = np.mean([np.mean(window[:third]), np.mean(window[-third:])])

        # Check if current frequency is an outlier
        if mean_log_psd[i] - center_power > coarseFreqDetectPowerDiff:
            noise_freq = freqs[i]
            noise_freqs.append(noise_freq)

            # Find end of the peak
            while (i < len(freqs) and 
                   mean_log_psd[i] - center_power > coarseFreqDetectLowerPowerDiff):
                i += 1
        else:
            i += 1

    return noise_freqs

def find_local_peaks(x, threshold):
    """
    Find local peaks in a 1D array that exceed a given threshold.
    
    Args:
    x (np.array): 1D input array
    threshold (float): Threshold for peak detection
    
    Returns:
    np.array: Indices of detected peaks
    """
    # Compute first order difference
    dx = np.diff(x)
    
    # Find where the derivative changes sign and exceeds the threshold
    peaks = np.where((np.hstack([dx, 0]) < 0) & 
                     (np.hstack([0, dx]) > 0) & 
                     (x > threshold))[0]
    
    return peaks