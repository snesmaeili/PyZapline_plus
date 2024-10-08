import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
class PyZaplinePlus:
    def __init__(self, data, sampling_rate, **kwargs):
        self.data = data
        self.sampling_rate = sampling_rate
        self.config = {
            'noisefreqs': kwargs.get('noisefreqs', []),
            'minfreq': kwargs.get('minfreq', 17),
            'maxfreq': kwargs.get('maxfreq', 99),
            'adaptiveNremove': kwargs.get('adaptiveNremove', True),
            'fixedNremove': kwargs.get('fixedNremove', 1),
            'detectionWinsize': kwargs.get('detectionWinsize', 6),
            'coarseFreqDetectPowerDiff': kwargs.get('coarseFreqDetectPowerDiff', 4),
            'coarseFreqDetectLowerPowerDiff': kwargs.get('coarseFreqDetectLowerPowerDiff', 1.76),
            'searchIndividualNoise': kwargs.get('searchIndividualNoise', True),
            'freqDetectMultFine': kwargs.get('freqDetectMultFine', 2),
            'detailedFreqBoundsUpper': kwargs.get('detailedFreqBoundsUpper', [-0.05, 0.05]),
            'detailedFreqBoundsLower': kwargs.get('detailedFreqBoundsLower', [-0.4, 0.1]),
            'maxProportionAboveUpper': kwargs.get('maxProportionAboveUpper', 0.005),
            'maxProportionBelowLower': kwargs.get('maxProportionBelowLower', 0.005),
            'noiseCompDetectSigma': kwargs.get('noiseCompDetectSigma', 3),
            'adaptiveSigma': kwargs.get('adaptiveSigma', True),
            'minsigma': kwargs.get('minsigma', 2.5),
            'maxsigma': kwargs.get('maxsigma', 5),  # Changed to 5
            'chunkLength': kwargs.get('chunkLength', 0),
            'minChunkLength': kwargs.get('minChunkLength', 30),
            'winSizeCompleteSpectrum': kwargs.get('winSizeCompleteSpectrum', sampling_rate * kwargs.get('chunkLength', 0)),
            'nkeep': kwargs.get('nkeep', 0),
            'plotResults': kwargs.get('plotResults', True),
            'segmentLength': kwargs.get('segmentLength', 1),
            'prominenceQuantile': kwargs.get('prominenceQuantile', 0.95),
            'overwritePlot': kwargs.get('overwritePlot', False),
            'figBase': kwargs.get('figBase', 100),
            'figPos': kwargs.get('figPos', None),
            'saveSpectra': kwargs.get('saveSpectra', False)
        }

    def finalize_inputs(self):
        """
        Finalize and prepare inputs for the Zapline-plus algorithm.
        """
        # Check and adjust sampling rate
        if self.sampling_rate > 500:
            print("WARNING: It is recommended to downsample the data to around 250Hz to 500Hz before applying Zapline-plus!")
            print(f"Current sampling rate is {self.sampling_rate}. Results may be suboptimal!")

        # Transpose data if necessary
        self.transpose_data = self.data.shape[1] > self.data.shape[0]
        if self.transpose_data:
            self.data = self.data.T

        # Adjust window size for spectrum calculation
        if self.config['winSizeCompleteSpectrum'] * self.sampling_rate > self.data.shape[0] / 8:
            self.config['winSizeCompleteSpectrum'] = np.floor(self.data.shape[0] / self.sampling_rate / 8)
            print("Data set is short. Adjusted window size for whole data set spectrum calculation to be 1/8 of the length!")

        # Set nkeep
        if self.config['nkeep'] == 0:
            self.config['nkeep'] = self.data.shape[1]
            
        # Detect flat channels
        self.flat_channels = self.detect_flat_channels()
        
        # Handle 'line' noise frequency
        if self.config['noisefreqs'] == 'line':
            self.detect_line_noise()

        # Compute initial spectrum
        self.pxx_raw_log, self.f = self.compute_spectrum(self.data)

        # Automatic noise frequency detection
        if not self.config['noisefreqs']:
            self.config['noisefreqs'] = self.detect_noise_frequencies()



    def detect_flat_channels(self):
        """
        Detect and return indices of flat channels in the data.
        """
        diff_data = np.diff(self.data, axis=0)
        flat_channels = np.where(np.all(diff_data == 0, axis=0))[0]
        if len(flat_channels) > 0:
            print(f"Flat channels detected (will be ignored and added back in after Zapline-plus processing): {flat_channels}")
            self.data = np.delete(self.data, flat_channels, axis=1)
        return flat_channels

    def compute_spectrum(self, data):
        """
        Compute the power spectral density of the input data.
        """
        f, pxx = signal.welch(data, fs=self.sampling_rate, 
                              nperseg=int(self.config['winSizeCompleteSpectrum'] * self.sampling_rate),
                              noverlap=0, axis=0)
        pxx_log = 10 * np.log10(pxx)
        
        if self.saveSpectra:
            self.analytics_resutls['f'] = f
            self.analytics_resutls['rawSpectrumLog'] = pxx_log
        return pxx_log, f
    
    def detect_line_noise(self):
        """
        Detect line noise (50 Hz or 60 Hz) in the data.
        """
        if self.config['noisefreqs'] != 'line':
            return
        idx = (self.f > 49) & (self.f < 51) | (self.f > 59) & (self.f < 61)
        spectra_chunk = self.pxx_raw_log[idx, :]
        max_val = np.max(spectra_chunk)
        freq_idx = np.unravel_index(np.argmax(spectra_chunk), spectra_chunk.shape)[0]
        noise_freq = self.f[idx][freq_idx]
        print(f"'noisefreqs' parameter was set to 'line', found line noise candidate at {noise_freq:.2f} Hz!")
        self.config['noisefreqs'] = []
        self.config['minfreq'] = noise_freq - self.config['detectionWinsize'] / 2
        self.config['maxfreq'] = noise_freq + self.config['detectionWinsize'] / 2
        return noise_freq
    
    def detect_noise_frequencies(self):
        """
        Automatically detect noise frequencies in the data.
        """
        noise_freqs = []
        current_minfreq = self.config['minfreq']
        
        while True:
            noisefreq, _, _, _ = find_next_noisefreq(
                self.pxx_raw_log,
                self.f,
                current_minfreq,
                self.config['coarseFreqDetectPowerDiff'],
                self.config['detectionWinsize'],
                self.config['maxfreq'],
                self.config['coarseFreqDetectLowerPowerDiff'],
                verbose=True
            )
            
            if noisefreq is None:
                break
            
            noise_freqs.append(noisefreq)
            current_minfreq = noisefreq + self.config['detectionWinsize'] / 2
            
            if current_minfreq >= self.config['maxfreq']:
                break
        
        return noise_freqs
    def fixed_chunk_detection(self):
        """
        Split the data into fixed-length chunks.
        """
        chunk_length_samples = int(self.config['chunkLength'] * self.sampling_rate)
        chunk_indices = [0]
        while chunk_indices[-1] + chunk_length_samples < len(self.data):
            chunk_indices.append(chunk_indices[-1] + chunk_length_samples)
        chunk_indices.append(len(self.data))
        return chunk_indices

    def adaptive_chunk_detection(self):
        """
        Use covariance matrices to adaptively segment data into chunks.
        """
        narrow_band_filtered = self.bandpass_filter(self.data, 
                                                    self.config['minfreq'], 
                                                    self.config['maxfreq'], 
                                                    self.sampling_rate)
        segment_length_samples = int(self.config['segmentLength'] * self.sampling_rate)
        n_segments = max(len(narrow_band_filtered) // segment_length_samples, 1)
        
        # Compute covariance matrices for each segment
        covariance_matrices = []
        for i in range(n_segments):
            start_idx = i * segment_length_samples
            end_idx = (i + 1) * segment_length_samples if i != n_segments - 1 else len(narrow_band_filtered)
            segment = narrow_band_filtered[start_idx:end_idx, :]
            cov_matrix = np.cov(segment, rowvar=False)
            covariance_matrices.append(cov_matrix)
        
        # Compute distances between consecutive covariance matrices
        distances = [
            np.sum(pdist(covariance_matrices[i] - covariance_matrices[i - 1])) / 2
            for i in range(1, len(covariance_matrices))
        ]
        
        # Find peaks in distances to determine chunk boundaries
        peaks, _ = find_peaks(distances, prominence=np.quantile(distances, self.config['prominenceQuantile']),
                              distance=self.config['minChunkLength'] * self.sampling_rate)
        chunk_indices = [0] + list((peaks + 1) * segment_length_samples) + [len(self.data)]
        return chunk_indices
    def detect_chunk_noise(self, chunk, noise_freq):
        """
        Detect noise frequency in a given chunk.
        """
        f, pxx_chunk = signal.welch(chunk, fs=self.sampling_rate, nperseg=len(chunk), axis=0)
        pxx_log = 10 * np.log10(pxx_chunk)
        
        freq_idx = (f > noise_freq - self.config['detectionWinsize'] / 2) & (f < noise_freq + self.config['detectionWinsize'] / 2)
        detailed_freq_idx = (f > noise_freq + self.config['detailedFreqBoundsUpper'][0]) & \
                            (f < noise_freq + self.config['detailedFreqBoundsUpper'][1])
        detailed_freqs = f[detailed_freq_idx]
        
        fine_data = np.mean(pxx_log[freq_idx, :], axis=1)
        third = len(fine_data) // 3
        center_data = np.mean([fine_data[:third], fine_data[-third:]])
        lower_quantile = np.mean([np.quantile(fine_data[:third], 0.05), np.quantile(fine_data[-third:], 0.05)])
        detailed_thresh = center_data + self.config['freqDetectMultFine'] * (center_data - lower_quantile)
        
        max_fine_power = np.max(np.mean(pxx_log[detailed_freq_idx, :], axis=1))
        if max_fine_power > detailed_thresh:
            return detailed_freqs[np.argmax(np.mean(pxx_log[detailed_freq_idx, :], axis=1))]
        return noise_freq

    def apply_zapline_to_chunk(self, chunk, noise_freq):
        """
        Apply noise removal to the chunk using DSS (Denoising Source Separation) based on the provided MATLAB code.
        """
        # Step 1: Apply smoothing to remove line frequency and harmonics
        smoothed_chunk = self.nt_smooth(chunk, 1 / noise_freq, self.config['adaptiveSigma'])

        # Step 2: PCA to reduce dimensionality and avoid overfitting
        truncated_chunk = self.nt_pca(smoothed_chunk, nkeep=self.config['nkeep'])

        # Step 3: DSS to isolate line components from residual
        n_harmonics = int(np.floor((0.5 * self.sampling_rate) / noise_freq))
        c0, c1 = self.nt_bias_fft(truncated_chunk, noise_freq * np.arange(1, n_harmonics + 1), self.config['winSizeCompleteSpectrum'])
        todss, pwr0, pwr1 = self.nt_dss0(c0, c1)
        scores = pwr1 / pwr0

        # Step 4: Determine the number of components to remove
        if self.config['adaptiveNremove']:
            adaptive_nremove, _ = self.iterative_outlier_removal(scores, self.config['noiseCompDetectSigma'])
            nremove = min(adaptive_nremove, len(scores) // 5)
        else:
            nremove = self.config['fixedNremove']

        # Step 5: Project the line-dominated components out of the original chunk
        if nremove > 0:
            line_components = self.nt_mmat(truncated_chunk, todss[:, :nremove])
            clean_chunk = chunk - line_components
        else:
            clean_chunk = chunk

        return clean_chunk, nremove, scores
    def compute_analytics(self, pxx_raw, pxx_clean, f, noise_freq):
        """
        Compute analytics to evaluate the cleaning process.
        """
        # Overall power removed (in log space)
        proportion_removed = 1 - 10 ** ((np.mean(pxx_clean) - np.mean(pxx_raw)) / 10)

        # Frequency range to evaluate below noise frequency
        freq_idx_below_noise = (f >= max(noise_freq - 11, 0)) & (f <= noise_freq - 1)
        proportion_removed_below_noise = (
            1 - 10 ** ((np.mean(pxx_clean[freq_idx_below_noise]) - np.mean(pxx_raw[freq_idx_below_noise])) / 10)
        )

        # Frequency range at noise frequency
        freq_idx_noise = (f > noise_freq + self.config['detailedFreqBoundsUpper'][0]) & \
                         (f < noise_freq + self.config['detailedFreqBoundsUpper'][1])
        proportion_removed_noise = (
            1 - 10 ** ((np.mean(pxx_clean[freq_idx_noise]) - np.mean(pxx_raw[freq_idx_noise])) / 10)
        )

        # Ratio of noise power to surroundings before and after cleaning
        freq_idx_noise_surrounding = (
            (f > noise_freq - (self.config['detectionWinsize'] / 2)) &
            (f < noise_freq - (self.config['detectionWinsize'] / 6))
        ) | (
            (f > noise_freq + (self.config['detectionWinsize'] / 6)) &
            (f < noise_freq + (self.config['detectionWinsize'] / 2))
        )

        ratio_noise_raw = 10 ** ((np.mean(pxx_raw[freq_idx_noise]) - np.mean(pxx_raw[freq_idx_noise_surrounding])) / 10)
        ratio_noise_clean = 10 ** ((np.mean(pxx_clean[freq_idx_noise]) - np.mean(pxx_clean[freq_idx_noise_surrounding])) / 10)

        return {
            'proportion_removed': proportion_removed,
            'proportion_removed_below_noise': proportion_removed_below_noise,
            'proportion_removed_noise': proportion_removed_noise,
            'ratio_noise_raw': ratio_noise_raw,
            'ratio_noise_clean': ratio_noise_clean
        }
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=0)        

    # def compute_analytics(self, pxx_raw, pxx_clean, f, noise_freq):
    #     pass

    def adaptive_cleaning(self, clean_data, raw_data, noise_freq):
        """
        Adjust the cleaning process if it was too weak or too strong.
        """
        f, pxx_clean = signal.welch(clean_data, fs=self.sampling_rate, 
                                    nperseg=int(self.config['winSizeCompleteSpectrum'] * self.sampling_rate),
                                    axis=0)
        pxx_clean_log = 10 * np.log10(pxx_clean)
        
        freq_idx_upper = (f > noise_freq + self.config['detailedFreqBoundsUpper'][0]) & \
                         (f < noise_freq + self.config['detailedFreqBoundsUpper'][1])
        freq_idx_lower = (f > noise_freq + self.config['detailedFreqBoundsLower'][0]) & \
                         (f < noise_freq + self.config['detailedFreqBoundsLower'][1])
        
        # Determine if the cleaning is too weak or too strong
        upper_thresh = np.mean(pxx_clean_log[freq_idx_upper, :])
        lower_thresh = np.mean(pxx_clean_log[freq_idx_lower, :])
        
        if upper_thresh > self.config['maxProportionAboveUpper']:
            # Cleaning was too weak, adjust
            self.config['noiseCompDetectSigma'] = min(self.config['noiseCompDetectSigma'] + 0.25, self.config['maxsigma'])
            return False, self.config
        elif lower_thresh < self.config['maxProportionBelowLower']:
            # Cleaning was too strong, adjust
            self.config['noiseCompDetectSigma'] = max(self.config['noiseCompDetectSigma'] - 0.25, self.config['minsigma'])
            return False, self.config
        
        return True, self.config
    # Helper methods to implement equivalent of MATLAB functions in Python
    def nt_smooth(self, x, T, n_iterations):
        """
        Smooth the data by convolution with a square window.
        """
        from scipy.ndimage import uniform_filter1d
        for _ in range(n_iterations):
            x = uniform_filter1d(x, size=int(T), axis=0)
        return x

    def nt_pca(self, x, nkeep):
        """
        Apply PCA and retain a specified number of components.
        """
        pca = PCA(n_components=nkeep)
        return pca.fit_transform(x)
    def nt_bias_fft(self, x, freq, nfft):
        """
        Compute covariance with and without filter bias.
        """
        from scipy.fft import fft
        filt = np.zeros(nfft // 2 + 1)
        idx = np.round(freq * nfft).astype(int)
        filt[idx] = 1
        filt = np.concatenate([filt, filt[-2:0:-1]])
        w = signal.windows.hann(nfft)
        c0 = np.cov(x, rowvar=False)
        c1 = np.zeros_like(c0)
        for i in range(0, len(x) - nfft, nfft // 2):
            z = x[i:i + nfft] * w[:, None]
            Z = fft(z, axis=0) * filt[:, None]
            c1 += np.real(np.dot(Z.T, Z))
        return c0, c1

    def nt_dss0(self, c0, c1):
        """
        Compute DSS from covariance matrices.
        """
        eigvals, eigvecs = np.linalg.eigh(np.dot(np.linalg.pinv(c0), c1))
        order = np.argsort(eigvals)[::-1]
        todss = eigvecs[:, order]
        pwr0 = np.sum((c0 @ todss) ** 2, axis=0)
        pwr1 = np.sum((c1 @ todss) ** 2, axis=0)
        return todss, pwr0, pwr1
    def nt_mmat(self, x, m):
        """
        Matrix multiplication (with convolution).
        """
        return np.dot(x, m)
    def iterative_outlier_removal(self, data_vector, sd_level=3):
        """
        Remove outliers in a vector based on an iterative sigma threshold approach.
        """
        threshold_old = np.max(data_vector)
        threshold = np.mean(data_vector) + sd_level * np.std(data_vector)
        n_remove = 0

        while threshold < threshold_old:
            flagged_points = data_vector > threshold
            data_vector = data_vector[~flagged_points]
            n_remove += np.sum(flagged_points)
            threshold_old = threshold
            threshold = np.mean(data_vector) + sd_level * np.std(data_vector)

        return n_remove, threshold  
    def add_back_flat_channels(self, clean_data):
        """
        Add back flat channels to the cleaned data.
        """
        if not self.flat_channels.size:
            return clean_data

        full_clean_data = np.zeros((clean_data.shape[0], clean_data.shape[1] + len(self.flat_channels)))

        # Insert back flat channels into their original positions
        for i, flat_chan in enumerate(self.flat_channels):
            full_clean_data[:, flat_chan] = 0  # Flat channel values are zero
        full_clean_data[:, [i for i in range(full_clean_data.shape[1]) if i not in self.flat_channels]] = clean_data

        return full_clean_data    
    def generate_output_figures(self, data, clean_data, noise_freq, zapline_config):
        """
        Generate figures to visualize the results.
        """
        f, pxx_raw = signal.welch(data, fs=self.sampling_rate, nperseg=int(self.config['winSizeCompleteSpectrum'] * self.sampling_rate), axis=0)
        f, pxx_clean = signal.welch(clean_data, fs=self.sampling_rate, nperseg=int(self.config['winSizeCompleteSpectrum'] * self.sampling_rate), axis=0)
        pxx_raw_log = 10 * np.log10(pxx_raw)
        pxx_clean_log = 10 * np.log10(pxx_clean)

        plt.figure(figsize=(20, 15))

        # Plot original power around noise frequency
        plt.subplot(3, 1, 1)
        this_freq_idx_plot = (f >= noise_freq - 1.1) & (f <= noise_freq + 1.1)
        plt.plot(f[this_freq_idx_plot], np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1), color='gray', label='Original Power')
        plt.title(f"Original Power Spectrum around {noise_freq} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (10*log10 μV^2/Hz)")
        plt.legend()
        plt.grid(True)

        # Plot number of components removed
        plt.subplot(3, 1, 2)
        plt.bar(range(len(zapline_config['nkeep'])), zapline_config['nkeep'], color='grey', alpha=0.5)
        plt.title(f"Number of Removed Components at {noise_freq} Hz")
        plt.xlabel("Chunk Index")
        plt.ylabel("Number of Components Removed")
        plt.grid(True)

        # Plot cleaned power around noise frequency
        plt.subplot(3, 1, 3)
        plt.plot(f[this_freq_idx_plot], np.mean(pxx_clean_log[this_freq_idx_plot, :], axis=1), color='green', label='Cleaned Power')
        plt.title(f"Cleaned Power Spectrum around {noise_freq} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (10*log10 μV^2/Hz)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    def run(self):
        
        self.finalize_inputs()
        
        clean_data = self.data.copy()
        zapline_config = self.config.copy()
        analytics_results = {}
        plot_handles = []

        for noise_freq in zapline_config['noisefreqs']:
            print(f"Removing noise at {noise_freq}Hz...")
            
            if self.config['chunkLength'] != 0:
                chunk_indices = self.fixed_chunk_detection()
            else:
                chunk_indices = self.adaptive_chunk_detection()
            
            n_chunks = len(chunk_indices) - 1
            print(f"{n_chunks} chunks will be created.")

            cleaning_done = False
            while not cleaning_done:
                scores = np.zeros((n_chunks, self.config['nkeep']))
                n_remove_final = np.zeros(n_chunks)
                noise_peaks = np.zeros(n_chunks)
                found_noise = np.zeros(n_chunks)

                for i_chunk in range(n_chunks):
                    chunk = clean_data[chunk_indices[i_chunk]:chunk_indices[i_chunk+1], :]
                    
                    if self.config['searchIndividualNoise']:
                        chunk_noise_freq = self.detect_chunk_noise(chunk, noise_freq)
                    else:
                        chunk_noise_freq = noise_freq

                    clean_chunk, n_remove, chunk_scores = self.apply_zapline_to_chunk(chunk, chunk_noise_freq)
                    clean_data[chunk_indices[i_chunk]:chunk_indices[i_chunk+1], :] = clean_chunk
                    
                    scores[i_chunk, :] = chunk_scores
                    n_remove_final[i_chunk] = n_remove
                    noise_peaks[i_chunk] = chunk_noise_freq
                    found_noise[i_chunk] = 1 if chunk_noise_freq != noise_freq else 0

                pxx_clean, f = self.compute_spectrum(clean_data)
                analytics = self.compute_analytics(self.pxx_raw_log, pxx_clean, self.f, noise_freq)
                
                cleaning_done, zapline_config = self.adaptive_cleaning(clean_data, self.data, noise_freq)

            if self.config['plotResults']:
                plot_handle = self.generate_output_figures(self.data, clean_data, noise_freq, zapline_config)
                plot_handles.append(plot_handle)

            analytics_results[f'noise_freq_{noise_freq}'] = {
                'scores': scores,
                'n_remove_final': n_remove_final,
                'noise_peaks': noise_peaks,
                'found_noise': found_noise,
                **analytics
            }

        if self.flat_channels:
            clean_data = self.add_back_flat_channels(clean_data)

        if self.transpose_data:
            clean_data = clean_data.T

        return clean_data, zapline_config, analytics_results, plot_handles  
def zapline_plus(data, sampling_rate, **kwargs):
    zp = PyZaplinePlus(data, sampling_rate, **kwargs)
    return zp.run()