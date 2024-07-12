import numpy as np
from scipy import signal

def apply_zapline(data, noise_freq, sampling_rate, n_remove=1):
    """
    Apply the Zapline algorithm to remove noise at a specific frequency.
    
    Args:
    data (np.array): Input data, shape (channels, samples)
    noise_freq (float): Frequency of the noise to be removed
    sampling_rate (float): Sampling rate of the data
    n_remove (int): Number of components to remove
    
    Returns:
    np.array: Cleaned data
    """
    # Compute the noise subspace
    n_samples = data.shape[1]
    t = np.arange(n_samples) / sampling_rate
    noise_subspace = np.column_stack([np.sin(2 * np.pi * noise_freq * t),
                                      np.cos(2 * np.pi * noise_freq * t)])
    
    # Project data onto noise subspace
    proj = np.dot(data, noise_subspace)
    
    # Compute PCA on projected data
    U, S, Vt = np.linalg.svd(proj, full_matrices=False)
    
    # Remove top components
    U[:, :n_remove] = 0
    S[:n_remove] = 0
    
    # Reconstruct cleaned data
    cleaned_proj = np.dot(U * S, Vt)
    cleaned_data = data - np.dot(cleaned_proj, noise_subspace.T)
    
    return cleaned_data

def detect_noise_components(data, noise_freq, sampling_rate, config):
    """
    Detect the number of noise components to remove.
    
    Args:
    data (np.array): Input data, shape (channels, samples)
    noise_freq (float): Frequency of the noise to be removed
    sampling_rate (float): Sampling rate of the data
    config (dict): Configuration parameters
    
    Returns:
    int: Number of components to remove
    """
    # Apply Zapline with a large number of components
    n_components = min(data.shape[0], 20)  # Use up to 20 components or number of channels
    cleaned_data = apply_zapline(data, noise_freq, sampling_rate, n_components)
    
    # Compute artifact scores
    f, psd_orig = signal.welch(data, fs=sampling_rate, nperseg=sampling_rate)
    f, psd_clean = signal.welch(cleaned_data, fs=sampling_rate, nperseg=sampling_rate)
    
    freq_mask = (f >= noise_freq - 0.5) & (f <= noise_freq + 0.5)
    artifact_scores = np.sum(psd_orig[:, freq_mask] - psd_clean[:, freq_mask], axis=1)
    
    # Detect outliers in artifact scores
    threshold = np.mean(artifact_scores) + config['noiseCompDetectSigma'] * np.std(artifact_scores)
    n_remove = np.sum(artifact_scores > threshold)
    
    return max(n_remove, config['fixedNremove'])

def adaptive_cleaning(data, noise_freq, sampling_rate, config):
    """
    Perform adaptive cleaning of the data.
    
    Args:
    data (np.array): Input data, shape (channels, samples)
    noise_freq (float): Frequency of the noise to be removed
    sampling_rate (float): Sampling rate of the data
    config (dict): Configuration parameters
    
    Returns:
    np.array: Cleaned data
    dict: Updated configuration
    """
    n_components = detect_noise_components(data, noise_freq, sampling_rate, config)
    cleaned_data = apply_zapline(data, noise_freq, sampling_rate, n_components)
    
    # Check if cleaning was too weak or too strong
    f, psd_clean = signal.welch(cleaned_data, fs=sampling_rate, nperseg=sampling_rate)
    freq_mask = (f >= noise_freq - 0.05) & (f <= noise_freq + 0.05)
    
    if np.any(psd_clean[:, freq_mask] > config['maxProportionAboveUpper']):
        config['noiseCompDetectSigma'] -= 0.25
        config['fixedNremove'] += 1
    elif np.any(psd_clean[:, freq_mask] < config['maxProportionBelowLower']):
        config['noiseCompDetectSigma'] += 0.25
        config['fixedNremove'] = max(config['fixedNremove'] - 1, 1)
    
    config['noiseCompDetectSigma'] = np.clip(config['noiseCompDetectSigma'], config['minsigma'], config['maxsigma'])
    
    return cleaned_data, config