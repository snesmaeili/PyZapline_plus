import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_spectrum(data, sampling_rate, title):
    """
    Plot the power spectrum of the data.
    
    Args:
    data (np.array): Input data, shape (channels, samples)
    sampling_rate (float): Sampling rate of the data
    title (str): Title for the plot
    """
    f, psd = signal.welch(data, fs=sampling_rate, nperseg=sampling_rate)
    plt.figure(figsize=(10, 6))
    plt.semilogy(f, psd.T, alpha=0.5)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.xlim(0, sampling_rate/2)
    plt.tight_layout()

def plot_time_series(data, sampling_rate, title):
    """
    Plot the time series data.
    
    Args:
    data (np.array): Input data, shape (channels, samples)
    sampling_rate (float): Sampling rate of the data
    title (str): Title for the plot
    """
    time = np.arange(data.shape[1]) / sampling_rate
    plt.figure(figsize=(12, 6))
    plt.plot(time, data.T, alpha=0.5)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

def plot_cleaning_results(original_data, cleaned_data, sampling_rate, noise_freq):
    """
    Plot the results of the cleaning process.
    
    Args:
    original_data (np.array): Original data, shape (channels, samples)
    cleaned_data (np.array): Cleaned data, shape (channels, samples)
    sampling_rate (float): Sampling rate of the data
    noise_freq (float): Frequency of the removed noise
    """
    plt.figure(figsize=(15, 10))
    
    # Original spectrum
    plt.subplot(2, 2, 1)
    plot_spectrum(original_data, sampling_rate, 'Original Spectrum')
    
    # Cleaned spectrum
    plt.subplot(2, 2, 2)
    plot_spectrum(cleaned_data, sampling_rate, 'Cleaned Spectrum')
    
    # Zoomed original spectrum
    plt.subplot(2, 2, 3)
    plot_spectrum(original_data, sampling_rate, f'Original Spectrum (Zoomed at {noise_freq} Hz)')
    plt.xlim(noise_freq - 5, noise_freq + 5)
    
    # Zoomed cleaned spectrum
    plt.subplot(2, 2, 4)
    plot_spectrum(cleaned_data, sampling_rate, f'Cleaned Spectrum (Zoomed at {noise_freq} Hz)')
    plt.xlim(noise_freq - 5, noise_freq + 5)
    
    plt.tight_layout()

def generate_output_figures(original_data, cleaned_data, noise_freq, config):
    """
    Generate and save output figures for the cleaning process.
    
    Args:
    original_data (np.array): Original data, shape (channels, samples)
    cleaned_data (np.array): Cleaned data, shape (channels, samples)
    noise_freq (float): Frequency of the removed noise
    config (dict): Configuration parameters
    """
    sampling_rate = config['sampling_rate']
    
    # Plot time series
    plot_time_series(original_data, sampling_rate, 'Original Time Series')
    plt.savefig('original_time_series.png')
    plt.close()
    
    plot_time_series(cleaned_data, sampling_rate, 'Cleaned Time Series')
    plt.savefig('cleaned_time_series.png')
    plt.close()
    
    # Plot cleaning results
    plot_cleaning_results(original_data, cleaned_data, sampling_rate, noise_freq)
    plt.savefig('cleaning_results.png')
    plt.close()
    
    # Additional analytics plots can be added here