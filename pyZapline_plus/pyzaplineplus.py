import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from noise_detection import find_next_noisefreq
from matplotlib import pyplot as plt
import numpy as np
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
            'winSizeCompleteSpectrum': kwargs.get('winSizeCompleteSpectrum', 300),
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
        if self.config['winSizeCompleteSpectrum'] * self.sampling_rate > self.data.shape[0]:
            self.config['winSizeCompleteSpectrum'] = np.floor(self.data.shape[0] / self.sampling_rate)
            print("Data set is short, results may be suboptimal!")

        # Set nkeep
        if self.config['nkeep'] == 0:
            self.config['nkeep'] = self.data.shape[1]
            
        # Detect flat channels
        self.flat_channels = self.detect_flat_channels()
        
        # Compute initial spectrum
        self.pxx_raw_log, self.f = self.compute_spectrum(self.data)

        # Handle 'line' noise frequency
        if self.config['noisefreqs'] == 'line':
            self.detect_line_noise()


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
        from scipy.signal import windows

        # Set window length and overlap to match MATLAB code
        nperseg = int(self.config['winSizeCompleteSpectrum'] * self.sampling_rate)
        noverlap = int(0.5 * nperseg)  # 50% overlap to match MATLAB's default behavior

        f, pxx = signal.welch(data, fs=self.sampling_rate,
                            window=windows.hann(nperseg),
                            nperseg=nperseg,
                            noverlap=noverlap, axis=0)

        # Log transform
        pxx_log = 10 * np.log10(pxx)
        
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
        # Ensure self.config has all necessary default parameters
        config_defaults = {
            'nfft': 128,
            'nkeep': None,
            'niterations': 1,
            'fig1': 100,
            'fig2': 101,
            'adaptiveNremove': 1,
            'noiseCompDetectSigma': 3,
            'fixedNremove': 1,  # Default nremove is 1
            'plotflag': False   # Set to True to enable plotting
        }
        for key, value in config_defaults.items():
            if key not in self.config or self.config[key] is None:
                self.config[key] = value

        # Step 1: Define line frequency normalized to sampling rate
        fline = noise_freq / self.sampling_rate

        # Check that fline is less than Nyquist frequency
        if fline >= 0.5:
            raise ValueError('fline should be less than Nyquist frequency (sampling_rate / 2)')

        # Step 2: Apply smoothing to remove line frequency and harmonics
        smoothed_chunk = self.nt_smooth(
            chunk,
            T=1 / fline,
            n_iterations=self.config['niterations']
        )

        # Step 3: Compute the residual after smoothing
        residual_chunk = chunk - smoothed_chunk

        # Step 4: PCA to reduce dimensionality and avoid overfitting
        if self.config['nkeep'] is None:
            self.config['nkeep'] = residual_chunk.shape[1]
        truncated_chunk, _ = self.nt_pca(
            residual_chunk,
            nkeep=self.config['nkeep']
        )

        # Step 5: DSS to isolate line components from residual
        n_harmonics = int(np.floor(0.5 / fline))
        harmonics = fline * np.arange(1, n_harmonics + 1)
        c0, c1 = self.nt_bias_fft(
            truncated_chunk,
            freq=harmonics,
            nfft=self.config['nfft']
        )
        todss, pwr0, pwr1 = self.nt_dss0(c0, c1)
        scores = pwr1 / pwr0

        # Optional plotting of DSS scores
        if self.config['plotflag']:
            plt.figure(self.config['fig1'])
            plt.clf()
            plt.plot(scores, '.-')
            plt.xlabel('Component')
            plt.ylabel('Score')
            plt.title('DSS to enhance line frequencies')
            # plt.savefig('dss_scores.png')

        # Step 6: Determine the number of components to remove
        if self.config['adaptiveNremove']:
            adaptive_nremove, _ = self.iterative_outlier_removal(
                scores,
                self.config['noiseCompDetectSigma']
            )
            if adaptive_nremove < self.config['fixedNremove']:
                nremove = self.config['fixedNremove']
                print(
                    f"Fixed nremove ({self.config['fixedNremove']}) is larger than adaptive nremove, using fixed nremove!"
                )
            else:
                nremove = adaptive_nremove
            if nremove > len(scores) // 5:
                nremove = len(scores) // 5
                print(
                    f"nremove is larger than 1/5th of the components, using that ({nremove})!"
                )
        else:
            nremove = self.config['fixedNremove']

        # Step 7: Project the line-dominated components out of the residual
        if nremove > 0:
            # Get line-dominated components
            line_components = self.nt_mmat(
                truncated_chunk,
                todss[:, :nremove]
            )
            # Project them out
            projected_chunk = self.nt_tsr(
                residual_chunk,
                line_components
            )
            # Reconstruct the clean signal
            clean_chunk = smoothed_chunk + projected_chunk
        else:
            clean_chunk = chunk

        # Optional plotting of spectra
        if self.config['plotflag']:
            # Normalize data for plotting
            norm_factor = np.sqrt(np.mean(chunk ** 2))
            chunk_norm = chunk / norm_factor
            clean_chunk_norm = clean_chunk / norm_factor
            removed_chunk_norm = (chunk - clean_chunk) / norm_factor

            # Compute spectra
            pxx_chunk, f = self.nt_spect_plot(
                chunk_norm,
                nfft=self.config['nfft'],
                fs=self.sampling_rate,
                return_data=True
            )
            pxx_clean, _ = self.nt_spect_plot(
                clean_chunk_norm,
                nfft=self.config['nfft'],
                fs=self.sampling_rate,
                return_data=True
            )
            pxx_removed, _ = self.nt_spect_plot(
                removed_chunk_norm,
                nfft=self.config['nfft'],
                fs=self.sampling_rate,
                return_data=True
            )

            divisor = np.sum(pxx_chunk, axis=0)
            # Plot original spectrum
            plt.figure(self.config['fig2'])
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.semilogy(f, np.abs(pxx_chunk) / divisor, label='Original', color='k')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Relative Power')
            plt.legend()
            plt.grid(True, which='both', axis='both')
            yl1 = plt.ylim()

            # Plot cleaned and removed spectra
            plt.subplot(1, 2, 2)
            plt.semilogy(f, np.abs(pxx_clean) / divisor, label='Clean', color='g')
            if nremove != 0:
                plt.semilogy(f, np.abs(pxx_removed) / divisor, label='Removed', color='r')
                plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.grid(True, which='both', axis='both')
            yl2 = plt.ylim()

            # Adjust y-limits to be the same
            yl_min = min(yl1[0], yl2[0])
            yl_max = max(yl1[1], yl2[1])
            plt.subplot(1, 2, 1)
            plt.ylim([yl_min, yl_max])
            plt.subplot(1, 2, 2)
            plt.ylim([yl_min, yl_max])

            plt.show()

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

        # Split T into integer and fractional parts
        integ = int(np.floor(T))
        frac = T - integ

        if integ >= x.shape[0]:
            # If window size exceeds data length, replace with mean
            x = np.tile(np.mean(x, axis=0), (x.shape[0], 1))
            return x

        # Remove onset step (similar to MATLAB code)
        mn = np.mean(x[:integ + 1, :], axis=0)
        x = x - mn

        # Smoothing iterations
        for _ in range(n_iterations):
            # Apply smoothing with integer part of the window size
            if integ > 0:
                x = uniform_filter1d(x, size=integ, axis=0)

            # Apply additional fractional smoothing if needed
            if frac > 0:
                # Smooth using fractional part (weighted average of neighbors)
                B = np.array([1 - frac, frac])
                x = np.apply_along_axis(lambda m: np.convolve(m, B, mode='same'), axis=0, arr=x)

        # Restore the mean value that was subtracted earlier
        x = x + mn

        return x



    def nt_pca(self, x, shifts=[0], nkeep=None, threshold=0, w=None):
        """
        Apply PCA with time shifts and retain a specified number of components.

        Parameters:
        - x: data matrix (n_samples, n_channels)
        - shifts: array of shifts to apply (default: [0])
        - nkeep: number of components to keep (default: all)
        - threshold: discard PCs with eigenvalues below this (default: 0)
        - w: weights (optional)

        Returns:
        - z: principal components
        - idx: indices of x that map to z
        """
        # Handling shifts and adjusting offsets
        offset = max(0, -min(shifts))
        shifts = [shift + offset for shift in shifts]  # Adjust shifts to make them nonnegative
        idx = offset + np.arange(x.shape[0] - max(shifts))  # x[idx] maps to z

        # Time-shifting the data
        x_shifted = self.nt_multishift(x, shifts)

        # Covariance calculation
        if w is not None:
            x_shifted = x_shifted * w[:, np.newaxis]
        cov_matrix = np.cov(x_shifted, rowvar=False)

        # PCA decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Apply threshold to eigenvalues
        if threshold > 0:
            valid_indices = eigvals > threshold
            eigvals = eigvals[valid_indices]
            eigvecs = eigvecs[:, valid_indices]

        # Retain specified number of components
        if nkeep is not None and nkeep < len(eigvals):
            eigvals = eigvals[:nkeep]
            eigvecs = eigvecs[:, :nkeep]

        # Project the shifted data onto the principal components
        z = np.dot(x_shifted, eigvecs)

        return z, idx

    def nt_multishift(self,x, shifts):
        x = np.asarray(x)
        shifts = np.asarray(shifts).flatten()
        nshifts = shifts.size

        if np.any(shifts < 0):
            raise ValueError('Shifts should be non-negative')
        if x.shape[0] < np.max(shifts):
            raise ValueError('Shifts should be no larger than the number of time samples in x')

        if x.ndim == 1:
            x = x[:, np.newaxis, np.newaxis]  # (time, 1, 1)
        elif x.ndim == 2:
            x = x[:, :, np.newaxis]  # (time, channels, 1)

        m, n, o = x.shape

        if nshifts == 1 and shifts[0] == 0:
            return x.squeeze()

        max_shift = np.max(shifts)
        N = m - max_shift

        # Initialize output array
        z = np.empty((N, n * nshifts, o), dtype=x.dtype)

        # Create shifted versions and concatenate
        for trial in range(o):
            x_trial = x[:, :, trial]  # (m, n)
            shifted_data = []
            for shift in shifts:
                shifted_slice = x_trial[shift:shift + N, :]
                shifted_data.append(shifted_slice)
            # Concatenate shifted data along the channel axis
            shifted_data = np.concatenate(shifted_data, axis=1)  # (N, n * nshifts)
            z[:, :, trial] = shifted_data

        return z.squeeze()


    def nt_cov(self, x, shifts, w=None):
        """
        Calculate time-shifted covariance of the data.

        Parameters:
        - x: data matrix (n_samples, n_channels)
        - shifts: array of shifts to apply
        - w: weights (optional)

        Returns:
        - c: covariance matrix
        - tw: total weight
        """
        nshifts = len(shifts)
        x_shifted = self.nt_multishift(x, shifts)

        if w is None:
            # No weights
            c = np.dot(x_shifted.T, x_shifted)
            tw = x_shifted.shape[0]
        else:
            # Apply weights
            w = w[:, np.newaxis] if w.ndim == 1 else w
            x_shifted = x_shifted * w
            c = np.dot(x_shifted.T, x_shifted)
            tw = np.sum(w)

        return c, tw

    def nt_pcarot(self, cov, nkeep=None, threshold=None, N=None):
        """
        Calculate PCA rotation matrix from covariance matrix.

        Parameters:
        - cov: covariance matrix
        - nkeep: number of components to keep (optional)
        - threshold: discard components below this threshold (optional)
        - N: eigs' K parameter (optional)

        Returns:
        - topcs: PCA rotation matrix
        - eigenvalues: PCA eigenvalues
        """
        if N is not None:
            eigvals, eigvecs = np.linalg.eigh(cov)
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)

        eigvecs = np.real(eigvecs)
        eigvals = np.real(eigvals)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Apply threshold to eigenvalues
        if threshold is not None:
            valid_indices = eigvals / eigvals[0] > threshold
            eigvals = eigvals[valid_indices]
            eigvecs = eigvecs[:, valid_indices]

        # Retain specified number of components
        if nkeep is not None:
            nkeep = min(nkeep, len(eigvals))
            eigvals = eigvals[:nkeep]
            eigvecs = eigvecs[:, :nkeep]

        return eigvecs, eigvals
    def nt_xcov(self,x, y, shifts=None, w=None):
        """
        Compute the cross-covariance of x and time-shifted y.

        Parameters:
            x (np.ndarray): Data array x (time x channels x trials).
            y (np.ndarray): Data array y (time x channels x trials).
            shifts (np.ndarray): Array of non-negative integer time shifts (default: [0]).
            w (np.ndarray): Optional weights array (time x 1 x trials) or (time x channels x trials).

        Returns:
            c (np.ndarray): Cross-covariance matrix.
            tw (float): Total weight.
        """
        if shifts is None:
            shifts = np.array([0])
        else:
            shifts = np.asarray(shifts).flatten()

        if np.any(shifts < 0):
            raise ValueError('Shifts must be non-negative integers')

        if w is not None and x.shape[0] != w.shape[0]:
            raise ValueError('x and w should have the same number of time samples')
        if x.shape[2] != y.shape[2]:
            raise ValueError('x and y should have the same number of trials')
        if w is not None and x.shape[2] != w.shape[2]:
            raise ValueError('x and w should have the same number of trials')

        nshifts = shifts.size
        mx, nx, ox = x.shape
        my, ny, oy = y.shape

        c = np.zeros((nx, ny * nshifts))

        if w is not None:
            x = x * w  # Broadcasting

        for k in range(ox):
            y_shifted = self.nt_multishift(y[:, :, k], shifts)  # Shape: (N, ny * nshifts)
            x_truncated = x[:y_shifted.shape[0], :, k]     # Truncate x to match shifted y
            c += x_truncated.T @ y_shifted  # Matrix multiplication

        if w is None:
            tw = ox * y_shifted.shape[0]
        else:
            w_truncated = w[:y_shifted.shape[0], :, :]
            tw = np.sum(w_truncated)

        return c, tw


    def nt_bias_fft(self, x, freq, nfft):
        from scipy.fft import fft
        from scipy.signal import windows

        """
        Compute covariance with and without filter bias.
        
        Parameters:
        - x: data matrix (n_samples, n_channels)
        - freq: row vector of normalized frequencies to keep (relative to sample rate)
        - nfft: FFT size

        Returns:
        - c0: unbiased covariance matrix
        - c1: biased covariance matrix
        """
        if np.max(freq) > 0.5:
            raise ValueError("Frequencies should be <= 0.5")
        if nfft > x.shape[0]:
            raise ValueError("nfft too large")

        filt = np.zeros(nfft // 2 + 1)

        if freq.ndim == 1:
            for k in range(freq.shape[0]):
                idx = int(round(freq[k] * nfft + 0.5))
                filt[idx] = 1
        elif freq.shape[0] == 2:
            for k in range(freq.shape[1]):
                idx = slice(int(round(freq[0, k] * nfft + 0.5)), int(round(freq[1, k] * nfft + 0.5)) + 1)
                filt[idx] = 1
        else:
            raise ValueError("freq should have one or two rows")

        filt = np.concatenate([filt, np.flipud(filt[1:-1])])
        w = windows.hann(nfft)

        c0 = np.cov(x, rowvar=False)
        c1 = np.zeros_like(c0)

        nframes = int(np.ceil((x.shape[0] - nfft / 2) / (nfft / 2)))
        for k in range(nframes):
            idx = int(k * nfft // 2)
            idx = min(idx, x.shape[0] - nfft)
            z = x[idx:idx + nfft, :]
            z = z * w[:, None]  # Apply Hann window
            Z = fft(z, axis=0) * filt[:, None]
            c1 += np.real(np.dot(Z.T, Z))

        return c0, c1
    def nt_dss0(self, c0, c1, keep1=None, keep2=10**-9):
        """
        Compute DSS from covariance matrices.
        
        Parameters:
        - c0: baseline covariance
        - c1: biased covariance
        - keep1: number of PCs to retain (default: all)
        - keep2: ignore PCs smaller than this threshold (default: 10^-9)

        Returns:
        - todss: matrix to convert data to normalized DSS components
        - pwr0: power per component (baseline)
        - pwr1: power per component (biased)
        """
        if c0.shape != c1.shape:
            raise ValueError("c0 and c1 should have the same size")
        if c0.shape[0] != c0.shape[1]:
            raise ValueError("c0 should be square")
        if np.any(np.isnan(c0)) or np.any(np.isnan(c1)):
            raise ValueError("NaN in covariance matrices")
        if np.any(np.isinf(c0)) or np.any(np.isinf(c1)):
            raise ValueError("INF in covariance matrices")

        # PCA and whitening matrix from the unbiased covariance
        topcs1, evs1 = self.nt_pcarot(c0, keep1, keep2)
        evs1 = np.abs(evs1)

        # Truncate PCA series if needed
        if keep1 is not None:
            topcs1 = topcs1[:, :keep1]
            evs1 = evs1[:keep1]
        if keep2 is not None:
            idx = evs1 / np.max(evs1) > keep2
            topcs1 = topcs1[:, idx]
            evs1 = evs1[idx]

        # Apply PCA and whitening to the biased covariance
        N = np.diag(1.0 / np.sqrt(evs1))
        c2 = N.T @ topcs1.T @ c1 @ topcs1 @ N

        # Matrix to convert PCA-whitened data to DSS
        topcs2, evs2 = self.nt_pcarot(c2, keep1, keep2)

        # DSS matrix (raw data to normalized DSS)
        todss = topcs1 @ N @ topcs2
        N2 = np.diag(todss.T @ c0 @ todss)
        todss = todss @ np.diag(1.0 / np.sqrt(N2))  # Adjust so that components are normalized

        # Power per DSS component
        pwr0 = np.sqrt(np.sum((c0 @ todss) ** 2, axis=0))  # Unbiased
        pwr1 = np.sqrt(np.sum((c1 @ todss) ** 2, axis=0))  # Biased

        return todss, pwr0, pwr1
    def nt_tsr(self,x, ref, shifts=None, wx=None, wref=None, keep=None, thresh=1e-20):
        """
        Perform time-shift regression (TSPCA) to denoise data.
        
        Parameters:
            x (np.ndarray): Data to denoise (time x channels x trials).
            ref (np.ndarray): Reference data (time x channels x trials).
            shifts (np.ndarray): Array of shifts to apply to ref (default: [0]).
            wx (np.ndarray): Weights to apply to x (time x 1 x trials).
            wref (np.ndarray): Weights to apply to ref (time x 1 x trials).
            keep (int): Number of shifted-ref PCs to retain (default: all).
            thresh (float): Threshold to ignore small shifted-ref PCs (default: 1e-20).
        
        Returns:
            y (np.ndarray): Denoised data.
            idx (np.ndarray): Indices where x(idx) is aligned with y.
            w (np.ndarray): Weights applied by tsr.
        """
        # Handle default arguments
        if shifts is None:
            shifts = np.array([0])
        else:
            shifts = np.asarray(shifts)
        
        if wx is not None and wx.ndim == 2:
            wx = wx[:, np.newaxis, :]
        if wref is not None and wref.ndim == 2:
            wref = wref[:, np.newaxis, :]
        
        # Check argument values for sanity
        if x.shape[0] != ref.shape[0]:
            raise ValueError('x and ref should have the same number of time samples')
        if x.shape[2] != ref.shape[2]:
            raise ValueError('x and ref should have the same number of trials')
        if wx is not None and (x.shape[0] != wx.shape[0] or x.shape[2] != wx.shape[2]):
            raise ValueError('x and wx should have matching dimensions')
        if wref is not None and (ref.shape[0] != wref.shape[0] or ref.shape[2] != wref.shape[2]):
            raise ValueError('ref and wref should have matching dimensions')
        if max(shifts) - min(0, min(shifts)) >= x.shape[0]:
            raise ValueError('x has too few samples to support the given shifts')
        if wx is not None and wx.shape[1] != 1:
            raise ValueError('wx should have shape (time, 1, trials)')
        if wref is not None and wref.shape[1] != 1:
            raise ValueError('wref should have shape (time, 1, trials)')
        if wx is not None and np.sum(wx) == 0:
            raise ValueError('weights on x are all zero!')
        if wref is not None and np.sum(wref) == 0:
            raise ValueError('weights on ref are all zero!')
        if shifts.size > 1000:
            raise ValueError(f'Number of shifts ({shifts.size}) is too large (if OK, adjust the code to allow it)')
        
        # Adjust x and ref to ensure that shifts are non-negative
        offset1 = max(0, -min(shifts))
        idx = np.arange(offset1, x.shape[0])
        x = x[idx, :, :]  # Truncate x
        if wx is not None:
            wx = wx[idx, :, :]
        shifts = shifts + offset1  # Shifts are now non-negative
        
        # Adjust size of x
        offset2 = max(0, max(shifts))
        idx_ref = np.arange(0, ref.shape[0] - offset2)
        x = x[:len(idx_ref), :, :]  # Part of x that overlaps with time-shifted refs
        if wx is not None:
            wx = wx[:len(idx_ref), :, :]
        
        mx, nx, ox = x.shape
        mref, nref, oref = ref.shape
        
        # Consolidate weights into a single weight matrix
        w = np.zeros((mx, 1, oref))
        if wx is None and wref is None:
            w[:] = 1
        elif wref is None:
            w = wx
        elif wx is None:
            for k in range(ox):
                wr = wref[:, :, k]
                wr_shifted = self.nt_multishift(wr, shifts)
                w[:, :, k] = np.min(wr_shifted, axis=1, keepdims=True)
        else:
            for k in range(ox):
                wr = wref[:, :, k]
                wr_shifted = self.nt_multishift(wr, shifts)
                w_min = np.min(wr_shifted, axis=1, keepdims=True)
                w[:, :, k] = np.minimum(w_min, wx[:w_min.shape[0], :, k])
        wx = w
        wref = np.zeros((mref, 1, oref))
        wref[idx, :, :] = w
        
        # Remove weighted means
        x0 = x.copy()
        x, _ = self.nt_demean(x, wx)
        mn1 = x - x0
        ref, _ = self.nt_demean(ref, wref)
        
        # Equalize power of ref channels, then equalize power of ref PCs
        ref = self.nt_normcol(ref, wref)
        ref = self.nt_pca(ref, thresh=1e-6)
        ref = self.nt_normcol(ref, wref)
        
        # Covariances and cross-covariance with time-shifted refs
        cref, twcref = self.nt_cov(ref, shifts, wref)
        cxref, twcxref = self.nt_xcov(x, ref, shifts, wx)
        
        # Regression matrix of x on time-shifted refs
        r = self.nt_regcov(cxref / twcxref, cref / twcref, keep=keep, thresh=thresh)
        
        # Clean x by removing regression on time-shifted refs
        y = np.zeros_like(x)
        for k in range(ox):
            ref_shifted = self.nt_multishift(ref[:, :, k], shifts)
            z = ref_shifted @ r
            y[:, :, k] = x[:z.shape[0], :, k] - z
        
        y0 = y.copy()
        y, _ = self.nt_demean(y, wx)  # Multishift(ref) is not necessarily zero mean
        mn2 = y - y0
        
        # idx for alignment
        idx_output = np.arange(offset1, offset1 + y.shape[0])
        mn = mn1 + mn2
        w = wref
        
        # Return outputs
        return y, idx_output, w    

    def nt_mmat(self,x, m):
        """
        Matrix multiplication (with convolution).

        Parameters:
            x (np.ndarray): Input data (can be 2D or multi-dimensional).
            m (np.ndarray): Matrix to apply.
                            - If 2D: Right multiply x by m.
                            - If 3D: Perform convolution-like operation along time.

        Returns:
            y (np.ndarray): Result after applying m to x.
        """
        # Handle the case where x is a list (similar to cell arrays in MATLAB)
        if isinstance(x, list):
            return [self.nt_mmat(xi, m) for xi in x]

        # Handle multi-dimensional x beyond 3D
        if x.ndim > 3:
            # Reshape x to 3D (time x channels x combined other dimensions)
            original_shape = x.shape
            time_dim = original_shape[0]
            chan_dim = original_shape[1]
            other_dims = original_shape[2:]
            x = x.reshape(time_dim, chan_dim, -1)
            y = self.nt_mmat(x, m)
            # Reshape y back to original dimensions
            y_shape = (y.shape[0], y.shape[1]) + other_dims
            y = y.reshape(y_shape)
            return y

        # If m is 2D, perform simple matrix multiplication
        if m.ndim == 2:
            # Unfold x to 2D
            if x.ndim == 3:
                time_dim, chan_dim, trial_dim = x.shape
                x_unfolded = x.reshape(time_dim * trial_dim, chan_dim)
            elif x.ndim == 2:
                x_unfolded = x
                time_dim, chan_dim = x.shape
                trial_dim = 1
            else:
                raise ValueError('x must be 2D or 3D array.')

            # Perform matrix multiplication
            x_multiplied = x_unfolded @ m

            # Fold back to original dimensions
            if trial_dim > 1:
                y = x_multiplied.reshape(time_dim, trial_dim, -1).transpose(0, 2, 1)
            else:
                y = x_multiplied

            return y

        else:
            # Convolution-like operation when m is 3D
            n_rows, n_cols, n_lags = m.shape

            if x.ndim == 2:
                x = x[:, :, np.newaxis]  # Add trial dimension
            n_samples, n_chans, n_trials = x.shape

            if n_chans != n_rows:
                raise ValueError("Number of channels in x must match number of rows in m.")

            # Initialize output array
            y = np.zeros((n_samples + n_lags - 1, n_cols, n_trials))

            # Perform convolution-like operation
            for trial in range(n_trials):
                x_trial = x[:, :, trial]  # Shape: (n_samples, n_chans)
                y_trial = np.zeros((n_samples + n_lags - 1, n_cols))

                # Unfold x_trial to 2D
                x_unfolded = x_trial  # Shape: (n_samples, n_chans)

                for lag in range(n_lags):
                    m_lag = m[:, :, lag]  # Shape: (n_rows, n_cols)
                    # Shift x_trial by lag
                    x_shifted = np.zeros((n_samples + n_lags - 1, n_chans))
                    x_shifted[lag:lag + n_samples, :] = x_unfolded

                    # Multiply and accumulate
                    y_partial = x_shifted @ m_lag  # Shape: (n_samples + n_lags - 1, n_cols)
                    y_trial += y_partial

                y[:, :, trial] = y_trial

            # If trials dimension was added artificially, remove it
            if n_trials == 1:
                y = y.squeeze(-1)  # Remove the last dimension if singleton

            return y


    def nt_demean(self, x, w=None):
        """
        Remove weighted mean over columns.

        Parameters:
            x (np.ndarray): Data array (time x channels x trials).
            w (np.ndarray): Optional weights array (time x 1 x trials) or (time x channels x trials).

        Returns:
            x_demeaned (np.ndarray): Demeaned data array.
            mn (np.ndarray): Mean values that were removed.
        """
        if w is not None and w.size < x.shape[0]:
            # Interpret w as array of indices to set to 1
            w_indices = w.flatten()
            if np.min(w_indices) < 0 or np.max(w_indices) >= x.shape[0]:
                raise ValueError('w interpreted as indices but values are out of range')
            w_full = np.zeros((x.shape[0], 1, x.shape[2]))
            w_full[w_indices, :, :] = 1
            w = w_full

        if w is not None and w.shape[2] != x.shape[2]:
            if w.shape[2] == 1 and x.shape[2] != 1:
                w = np.tile(w, (1, 1, x.shape[2]))
            else:
                raise ValueError('w should have same number of trials as x, or be singleton in that dimension')

        m, n, o = x.shape
        x_unfolded = x.reshape(m, -1)  # Unfold x to 2D array (time x (channels*trials))

        if w is None:
            # Unweighted mean
            mn = np.mean(x_unfolded, axis=0)
            x_demeaned = x_unfolded - mn
        else:
            w_unfolded = w.reshape(m, -1)
            if w_unfolded.shape[0] != x_unfolded.shape[0]:
                raise ValueError('x and w should have the same number of time samples')

            sum_w = np.sum(w_unfolded, axis=0, keepdims=True) + np.finfo(float).eps
            mn = np.sum(x_unfolded * w_unfolded, axis=0, keepdims=True) / sum_w
            x_demeaned = x_unfolded - mn

        x_demeaned = x_demeaned.reshape(m, n, o)
        mn = mn.reshape(1, n, o)

        return x_demeaned, mn
    def nt_normcol(self, x, w=None):
        """
        Normalize each column so its weighted mean square is 1.

        Parameters:
            x (np.ndarray): Data array (time x channels x trials).
            w (np.ndarray): Optional weights array with same dimensions as x or (time x 1 x trials).

        Returns:
            y (np.ndarray): Normalized data array.
            norms (np.ndarray): Vector of norms used for normalization.
        """
        if x.size == 0:
            raise ValueError('Empty x')

        if isinstance(x, list):
            raise NotImplementedError('Weights not supported for list inputs')

        if x.ndim == 4:
            # Apply normcol to each "book" (4th dimension)
            m, n, o, p = x.shape
            y = np.zeros_like(x)
            N = np.zeros(n)
            for k in range(p):
                y[:, :, :, k], NN = self.nt_normcol(x[:, :, :, k])
                N += NN ** 2
            return y, np.sqrt(N)

        if x.ndim == 3:
            # Unfold data to 2D
            m, n, o = x.shape
            x_unfolded = x.reshape(m * o, n)
            if w is None:
                y_unfolded, N = self.nt_normcol(x_unfolded)
            else:
                if w.shape[0] != m:
                    raise ValueError('Weight matrix should have same number of time samples as data')
                if w.ndim == 2 and w.shape[1] == 1:
                    w = np.tile(w, (1, n, o))
                if w.shape != x.shape:
                    raise ValueError('Weight should have same shape as data')
                w_unfolded = w.reshape(m * o, n)
                y_unfolded, N = self.nt_normcol(x_unfolded, w_unfolded)
            y = y_unfolded.reshape(m, n, o)
            norms = np.sqrt(N)
            return y, norms

        elif x.ndim == 2:
            # 2D data
            m, n = x.shape
            if w is None:
                # No weight
                N = np.sum(x ** 2, axis=0) / m
                N_inv_sqrt = np.where(N == 0, 0, 1.0 / np.sqrt(N))
                y = x * N_inv_sqrt
            else:
                if w.shape[0] != x.shape[0]:
                    raise ValueError('Weight matrix should have same number of time samples as data')
                if w.ndim == 1 or (w.ndim == 2 and w.shape[1] == 1):
                    w = np.tile(w.reshape(-1, 1), (1, n))
                if w.shape != x.shape:
                    raise ValueError('Weight should have same shape as data')
                sum_w = np.sum(w, axis=0)
                N = np.sum((x ** 2) * w, axis=0) / (sum_w + np.finfo(float).eps)
                N_inv_sqrt = np.where(N == 0, 0, 1.0 / np.sqrt(N))
                y = x * N_inv_sqrt
            norms = np.sqrt(N)
            return y, norms
        else:
            raise ValueError('Input data must be 2D or 3D')
    def nt_regcov(self, cxy, cyy, keep=None, threshold=0):
        """
        Compute regression matrix from cross-covariance matrices.

        Parameters:
            cxy (np.ndarray): Cross-covariance matrix between data and regressor.
            cyy (np.ndarray): Covariance matrix of regressor.
            keep (int): Number of regressor PCs to keep (default: all).
            threshold (float): Eigenvalue threshold for discarding regressor PCs (default: 0).

        Returns:
            r (np.ndarray): Regression matrix to apply to regressor to best model data.
        """
        # PCA of regressor covariance matrix
        eigenvalues, topcs = np.linalg.eigh(cyy)
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        topcs = topcs[:, idx]

        # Discard negligible regressor PCs
        if keep is not None:
            keep = min(keep, topcs.shape[1])
            topcs = topcs[:, :keep]
            eigenvalues = eigenvalues[:keep]

        if threshold is not None and threshold > 0:
            idx_thresh = eigenvalues / np.max(eigenvalues) > threshold
            topcs = topcs[:, idx_thresh]
            eigenvalues = eigenvalues[idx_thresh]

        # Cross-covariance between data and regressor PCs
        cxy = cxy.T  # Transpose cxy to match dimensions
        r = topcs.T @ cxy

        # Projection matrix from regressor PCs
        r = r / eigenvalues[:, np.newaxis]

        # Projection matrix from regressors
        r = topcs @ r

        return r
    def nt_spect_plot(self, data, nfft, fs, return_data=False):
        """
        Compute and optionally plot the power spectrum of the data.
        """
        f, pxx = signal.welch(data, fs=fs, nperseg=nfft, axis=0)
        if return_data:
            return pxx, f
        else:
            plt.figure()
            plt.semilogy(f, np.mean(pxx, axis=1))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.title('Power Spectrum')
            plt.grid(True)
            plt.show()


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

        fig, axes = plt.subplots(3, 1, figsize=(20, 15))

        # Plot original power around noise frequency
        this_freq_idx_plot = (f >= noise_freq - 1.1) & (f <= noise_freq + 1.1)
        axes[0].plot(f[this_freq_idx_plot], np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1), color='gray', label='Original Power')
        axes[0].set_title(f"Original Power Spectrum around {noise_freq} Hz")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Power (10*log10 μV^2/Hz)")
        axes[0].legend()
        axes[0].grid(True)

        # Plot number of components removed
        axes[1].bar(range(len(zapline_config['nkeep'])), zapline_config['nkeep'], color='grey', alpha=0.5)
        axes[1].set_title(f"Number of Removed Components at {noise_freq} Hz")
        axes[1].set_xlabel("Chunk Index")
        axes[1].set_ylabel("Number of Components Removed")
        axes[1].grid(True)

        # Plot cleaned power around noise frequency
        axes[2].plot(f[this_freq_idx_plot], np.mean(pxx_clean_log[this_freq_idx_plot, :], axis=1), color='green', label='Cleaned Power')
        axes[2].set_title(f"Cleaned Power Spectrum around {noise_freq} Hz")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Power (10*log10 μV^2/Hz)")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        return fig
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
                analytics = self.compute_analytics(self.pxx_raw_log, pxx_clean, f, noise_freq)
                
                cleaning_done, zapline_config = self.adaptive_cleaning(clean_data, self.data, noise_freq)

            if self.config['plotResults']:
                plot_handle = self.generate_output_figures(self.data, clean_data, noise_freq, zapline_config)
                if plot_handle:
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