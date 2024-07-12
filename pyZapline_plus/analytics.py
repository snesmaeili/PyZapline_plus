import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class ZaplineAnalytics:
    def __init__(self, data, sampling_rate, config):
        self.data = data
        self.sampling_rate = sampling_rate
        self.config = config
        self.analytics = {}

    def compute_analytics(self, cleaned_data, noise_freqs, removed_components, artifact_scores, chunk_noise_peaks):
        self.analytics['raw_spectrum'] = self.compute_spectrum(self.data)
        self.analytics['cleaned_spectrum'] = self.compute_spectrum(cleaned_data)
        self.analytics['noise_freqs'] = noise_freqs
        self.analytics['removed_components'] = removed_components
        self.analytics['artifact_scores'] = artifact_scores
        self.analytics['chunk_noise_peaks'] = chunk_noise_peaks
        
        for freq in noise_freqs:
            self.analytics[f'power_ratio_{freq}Hz'] = self.compute_power_ratio(freq)
            self.analytics[f'removed_power_{freq}Hz'] = self.compute_removed_power(freq)
            self.analytics[f'threshold_proportions_{freq}Hz'] = self.compute_threshold_proportions(freq)

    def compute_spectrum(self, data):
        f, psd = signal.welch(data, fs=self.sampling_rate, nperseg=self.sampling_rate)
        return f, 10 * np.log10(psd)

    def compute_power_ratio(self, freq):
        f, psd = self.analytics['raw_spectrum']
        f_clean, psd_clean = self.analytics['cleaned_spectrum']
        
        freq_mask = (f >= freq - 0.05) & (f <= freq + 0.05)
        noise_power_raw = np.mean(psd[:, freq_mask])
        noise_power_clean = np.mean(psd_clean[:, freq_mask])
        
        surrounding_mask = ((f >= freq - 1) & (f < freq - 0.05)) | ((f > freq + 0.05) & (f <= freq + 1))
        surrounding_power_raw = np.mean(psd[:, surrounding_mask])
        surrounding_power_clean = np.mean(psd_clean[:, surrounding_mask])
        
        return {
            'before': noise_power_raw / surrounding_power_raw,
            'after': noise_power_clean / surrounding_power_clean
        }

    def compute_removed_power(self, freq):
        f, psd = self.analytics['raw_spectrum']
        f_clean, psd_clean = self.analytics['cleaned_spectrum']
        
        freq_mask = (f >= freq - 0.05) & (f <= freq + 0.05)
        power_raw = np.mean(psd[:, freq_mask])
        power_clean = np.mean(psd_clean[:, freq_mask])
        
        return (power_raw - power_clean) / power_raw

    def compute_threshold_proportions(self, freq):
        f, psd = self.analytics['cleaned_spectrum']
        
        upper_bound = self.config['detailedFreqBoundsUpper']
        lower_bound = self.config['detailedFreqBoundsLower']
        
        upper_mask = (f >= freq + upper_bound[0]) & (f <= freq + upper_bound[1])
        lower_mask = (f >= freq + lower_bound[0]) & (f <= freq + lower_bound[1])
        
        center_power = np.mean(psd[:, (f >= freq - 1) & (f <= freq + 1)])
        threshold = center_power + 2 * self.config['freqDetectMultFine']
        
        above_upper = np.mean(psd[:, upper_mask] > threshold)
        below_lower = np.mean(psd[:, lower_mask] < threshold)
        
        return {
            'above_upper': above_upper,
            'below_lower': below_lower
        }

    def generate_plots(self):
        for freq in self.analytics['noise_freqs']:
            self.plot_frequency_cleaning(freq)

    def plot_frequency_cleaning(self, freq):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Zapline-plus Cleaning Results for {freq} Hz')

        self.plot_spectrum(axs[0, 0], freq)
        self.plot_removed_components(axs[0, 1])
        self.plot_chunk_noise_peaks(axs[0, 2])
        self.plot_artifact_scores(axs[0, 3])
        self.plot_cleaned_spectrum(axs[1, 0], freq)
        self.plot_full_spectrum(axs[1, 1])
        self.plot_clean_vs_noise(axs[1, 2], freq)
        self.plot_below_noise(axs[1, 3], freq)

        plt.tight_layout()
        plt.savefig(f'zapline_plus_cleaning_{freq}Hz.png')
        plt.close()

    def plot_spectrum(self, ax, freq):
        f, psd = self.analytics['raw_spectrum']
        ax.semilogy(f, psd.T)
        ax.set_xlim(freq - 1.1, freq + 1.1)
        ax.set_title('Raw Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')

    def plot_removed_components(self, ax):
        ax.plot(self.analytics['removed_components'])
        ax.set_title('Removed Components per Chunk')
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Number of Components')

    def plot_chunk_noise_peaks(self, ax):
        ax.plot(self.analytics['chunk_noise_peaks'])
        ax.set_title('Detected Noise Peaks per Chunk')
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Frequency (Hz)')

    def plot_artifact_scores(self, ax):
        ax.plot(self.analytics['artifact_scores'])
        ax.set_title('Artifact Scores')
        ax.set_xlabel('Component')
        ax.set_ylabel('Score')

    def plot_cleaned_spectrum(self, ax, freq):
        f, psd = self.analytics['cleaned_spectrum']
        ax.semilogy(f, psd.T)
        ax.set_xlim(freq - 1.1, freq + 1.1)
        ax.set_title('Cleaned Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')

    def plot_full_spectrum(self, ax):
        f, psd = self.analytics['raw_spectrum']
        f_clean, psd_clean = self.analytics['cleaned_spectrum']
        ax.semilogy(f, psd.T, alpha=0.5, label='Raw')
        ax.semilogy(f_clean, psd_clean.T, alpha=0.5, label='Cleaned')
        ax.set_title('Full Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.legend()

    def plot_clean_vs_noise(self, ax, freq):
        f, psd = self.analytics['raw_spectrum']
        f_clean, psd_clean = self.analytics['cleaned_spectrum']
        noise_psd = psd - psd_clean
        ax.semilogy(f - freq, psd_clean.T, label='Cleaned')
        ax.semilogy(f - freq, noise_psd.T, label='Removed Noise')
        ax.set_title('Cleaned vs. Removed Noise')
        ax.set_xlabel('Relative Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.legend()

    def plot_below_noise(self, ax, freq):
        f, psd = self.analytics['raw_spectrum']
        f_clean, psd_clean = self.analytics['cleaned_spectrum']
        mask = (f >= freq - 10) & (f < freq)
        ax.semilogy(f[mask], psd[:, mask].T, label='Raw')
        ax.semilogy(f_clean[mask], psd_clean[:, mask].T, label='Cleaned')
        ax.set_title('Spectrum Below Noise Frequency')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.legend()