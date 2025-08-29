import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from typing import List, Optional, Tuple, Union, cast
from .noise_detection import find_next_noisefreq
class PyZaplinePlus:
    def __init__(self, data, sampling_rate, **kwargs):
        # Validate inputs
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # Check for empty data
        if data.size == 0:
            raise ValueError("Data array cannot be empty")
            
        # Validate sampling rate
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number")
            
        # Ensure data is 2D (samples x channels)
        if data.ndim == 1:
            data = data.reshape(-1, 1)  # Convert to column vector
        elif data.ndim > 2:
            raise ValueError("Data must be 1D or 2D array")
            
        self.data = data
        self.sampling_rate = sampling_rate
        self.config = {
            'noisefreqs': kwargs.get('noisefreqs', []),
            'minfreq': kwargs.get('minfreq', 17),
            'maxfreq': kwargs.get('maxfreq', 99),
            'adaptiveNremove': kwargs.get('adaptiveNremove', True),
            'fixedNremove': kwargs.get('fixedNremove', 1),
            'detectionWinsize': kwargs.get('detectionWinsize', 6),
            # Match MATLAB default: 4 (2.5x power over mean in dB scale)
            'coarseFreqDetectPowerDiff': kwargs.get('coarseFreqDetectPowerDiff', 4),
            'coarseFreqDetectLowerPowerDiff': kwargs.get('coarseFreqDetectLowerPowerDiff', 1.76091259055681),
            'searchIndividualNoise': kwargs.get('searchIndividualNoise', True),
            'freqDetectMultFine': kwargs.get('freqDetectMultFine', 2.0),
            'detailedFreqBoundsUpper': kwargs.get('detailedFreqBoundsUpper', [-0.05, 0.05]),
            'detailedFreqBoundsLower': kwargs.get('detailedFreqBoundsLower', [-0.4, 0.1]),
            'maxProportionAboveUpper': kwargs.get('maxProportionAboveUpper', 0.005),
            'maxProportionBelowLower': kwargs.get('maxProportionBelowLower', 0.005),
            'noiseCompDetectSigma': kwargs.get('noiseCompDetectSigma', 3.0),
            'adaptiveSigma': kwargs.get('adaptiveSigma', True),
            'minsigma': kwargs.get('minsigma', 2.5),
            'maxsigma': kwargs.get('maxsigma', 5.0),  # Changed to 5
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

        # Adjust window size for spectrum calculation to mirror MATLAB behavior
        # MATLAB ensures at least 8 segments for pwelch by setting the window to ~1/8 of data length
        if self.config['winSizeCompleteSpectrum'] * self.sampling_rate > (self.data.shape[0] / 8):
            new_win = int(np.floor((self.data.shape[0] / self.sampling_rate) / 8))
            self.config['winSizeCompleteSpectrum'] = max(new_win, 1)
            print('Data set is short. Adjusted window size for whole data set spectrum calculation to be 1/8 of the length!')

        # Set nkeep
        if self.config['nkeep'] == 0:
            self.config['nkeep'] = self.data.shape[1]

        # Track the minimum allowed fixedNremove to support adaptive updates
        if 'baseFixedNremove' not in self.config:
            self.config['baseFixedNremove'] = int(self.config.get('fixedNremove', 1))
            
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
        
        # Ensure nperseg doesn't exceed data length
        max_nperseg = data.shape[0]
        if nperseg > max_nperseg:
            nperseg = max_nperseg
            
        # Ensure minimum nperseg for meaningful spectrum
        if nperseg < 8:
            nperseg = min(8, max_nperseg)
            
        noverlap = int(0.5 * nperseg)  # 50% overlap to match MATLAB's default behavior

        f, pxx = signal.welch(data,
                            fs=self.sampling_rate,
                            window=windows.hann(nperseg,sym=True),
                            nperseg=nperseg,
                            noverlap=noverlap, axis=0)

        # Log transform
        pxx_log = 10 * np.log10(pxx)
        
        return pxx_log, f
        
    def detect_line_noise(self):
        """
        Detect line noise (50 Hz or 60 Hz) in the data.

        Notes:
        - The MATLAB implementation searches within narrow bands around 50/60 Hz.
          To robustly capture the true peak (which might fall exactly on-bin), we
          use a slightly wider 2 Hz window: 49–51 Hz and 59–61 Hz.
        """
        if self.config['noisefreqs'] != 'line':
            return
        # Use a robust 2 Hz window around 50 Hz and 60 Hz
        idx = ((self.f > 49) & (self.f < 51)) | ((self.f > 59) & (self.f < 61))
        if not np.any(idx):
            # As a fallback, try the original narrow bounds
            idx = ((self.f > 49) & (self.f < 50)) | ((self.f > 59) & (self.f < 60))
        spectra_chunk = self.pxx_raw_log[idx, :]
        # Flatten across channels and pick global maximum within the search band
        flat_idx = np.argmax(spectra_chunk)
        row_idx = np.unravel_index(flat_idx, spectra_chunk.shape)[0]
        noise_freq = self.f[idx][row_idx]
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
        threshs=[]
        current_minfreq = self.config['minfreq']
        self.config['automaticFreqDetection'] = True
        while True:
            noisefreq, _, _, thresh = find_next_noisefreq(
                self.pxx_raw_log,
                self.f,
                current_minfreq,
                self.config['coarseFreqDetectPowerDiff'],
                self.config['detectionWinsize'],
                self.config['maxfreq'],
                self.config['coarseFreqDetectLowerPowerDiff'],
                verbose=False
            )
            
            if noisefreq is None:
                break
            
            noise_freqs.append(noisefreq)
            threshs.append(thresh)
            current_minfreq = noisefreq + self.config['detectionWinsize'] / 2
            
            if current_minfreq >= self.config['maxfreq']:
                break
        self.config['thresh'] = threshs
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



    def adaptive_chunk_detection(self,noise_freq):
        """
        Use covariance matrices to adaptively segment data into chunks.
        """
        from scipy.signal import find_peaks
        from scipy.spatial.distance import pdist
        # 1. Bandpass Filter the Data
        narrow_band_filtered = self.bandpass_filter(
            self.data,
            noise_freq - self.config['detectionWinsize'] / 2,
            noise_freq + self.config['detectionWinsize'] / 2,
            self.sampling_rate,
        )
        
        # 2. Determine Segment Length and Number of Segments
        segment_length_samples = int(self.config['segmentLength'] * self.sampling_rate)
        n_segments = max(len(narrow_band_filtered) // segment_length_samples, 1)
        
        # 3. Compute Covariance Matrices for Each Segment
        covariance_matrices = []
        for i in range(n_segments):
            start_idx = i * segment_length_samples
            end_idx = (i + 1) * segment_length_samples if i != n_segments - 1 else len(narrow_band_filtered)
            segment = narrow_band_filtered[start_idx:end_idx, :]
            cov_matrix = np.cov(segment, rowvar=False)
            covariance_matrices.append(cov_matrix)
        
        # 4. Compute Distances Between Consecutive Covariance Matrices (match MATLAB)
        # distances(i-1) = sum(pdist(C_i - C_{i-1})) / 2
        distances = []
        for i in range(1, len(covariance_matrices)):
            cov_diff = covariance_matrices[i] - covariance_matrices[i - 1]
            # Sum of pairwise distances across rows
            d = pdist(cov_diff)
            distance = np.sum(d) / 2.0
            distances.append(distance)
        distances = np.array(distances)
        
        # 5. First Find Peaks to Obtain Prominences
        initial_peaks, properties = find_peaks(distances,prominence=0)
        prominences = properties['prominences']
        
        # 6. Determine Prominence Threshold Based on Quantile
        if len(prominences) == 0:
            prominence_threshold: float = float('inf')  # No peaks found
        else:
            prominence_threshold: float = float(np.quantile(prominences, self.config['prominenceQuantile']))
        
        # 7. Second Find Peaks Using Prominence Threshold
        min_peak_distance_segments = int(np.ceil(self.config['minChunkLength'] / self.config['segmentLength']))
        peaks, _ = find_peaks(
            distances,
            prominence=prominence_threshold,
            distance=min_peak_distance_segments
        )
        # 8. Create Final Chunk Indices
        # Initialize with 0 (start of data)
        chunk_indices = [0]
        
        # Calculate the end indices of the peaks in terms of samples
        for peak in peaks:
            # peak is the index in 'distances', corresponding to the boundary between segments
            # So, the peak corresponds to the end of segment 'peak' and start of 'peak+1'
            end_sample = (peak + 1) * segment_length_samples
            chunk_indices.append(end_sample)
        
        # Append the end of data
        chunk_indices.append(len(self.data))
        
        # 9. Ensure All Chunks Meet Minimum Length
        min_length_samples = int(self.config['minChunkLength'] * self.sampling_rate)
        
        # Check the first and last chunk only if we have at least two boundaries
        if len(chunk_indices) > 2:
            if chunk_indices[1] - chunk_indices[0] < min_length_samples:
                chunk_indices.pop(1)  # Remove the first peak
        if len(chunk_indices) > 2:
            if chunk_indices[-1] - chunk_indices[-2] < min_length_samples:
                chunk_indices.pop(-2)  # Remove the last peak
        
        # Sort and remove duplicates if any
        chunk_indices = sorted(set(chunk_indices))
        
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
        
        max_fine_power: float = float(np.max(np.mean(pxx_log[detailed_freq_idx, :], axis=1)))
        if max_fine_power > detailed_thresh:
            return detailed_freqs[np.argmax(np.mean(pxx_log[detailed_freq_idx, :], axis=1))]
        return noise_freq

    def apply_zapline_to_chunk(self, chunk, noise_freq):
        """
        Apply noise removal to the chunk using DSS (Denoising Source Separation) based on the provided MATLAB code.
        """
        # Ensure self.config has all necessary default parameters
        config_defaults = {
            'nfft': 1024,
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
        if scores is None or len(scores) == 0:
            nremove = 0
        elif self.config['adaptiveNremove']:
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
            # Cap removal to at most 1/5 of components, but ensure at least 1 when any components exist
            if nremove > len(scores) // 5:
                nremove = max(1, len(scores) // 5)
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
            projected_chunk,_,_= self.nt_tsr(
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
    def compute_analytics(self, pxx_raw_log, pxx_clean_log, f, noise_freq):
        """
        Compute analytics to evaluate the cleaning process.
        """
        # Overall power removed (in log space)
        proportion_removed = 1 - 10 ** ((np.mean(pxx_clean_log) - np.mean(pxx_raw_log)) / 10)

        # Frequency range to evaluate below noise frequency
        freq_idx_below_noise = (f >= max(noise_freq - 11, 0)) & (f <= noise_freq - 1)
        proportion_removed_below_noise = 1 - 10 ** (
            (np.mean(pxx_clean_log[freq_idx_below_noise, :]) - np.mean(pxx_raw_log[freq_idx_below_noise, :])) / 10
        )

        # Frequency range at noise frequency
        freq_idx_noise = (f > noise_freq + self.config['detailedFreqBoundsUpper'][0]) & \
                        (f < noise_freq + self.config['detailedFreqBoundsUpper'][1])
        proportion_removed_noise = 1 - 10 ** (
            (np.mean(pxx_clean_log[freq_idx_noise, :]) - np.mean(pxx_raw_log[freq_idx_noise, :])) / 10
        )

        # Ratio of noise power to surroundings before and after cleaning
        freq_idx_noise_surrounding = (
            ((f > noise_freq - (self.config['detectionWinsize'] / 2)) & (f < noise_freq - (self.config['detectionWinsize'] / 6))) |
            ((f > noise_freq + (self.config['detectionWinsize'] / 6)) & (f < noise_freq + (self.config['detectionWinsize'] / 2)))
        )

        mean_pxx_raw_noise = np.mean(pxx_raw_log[freq_idx_noise, :])
        mean_pxx_raw_noise_surrounding = np.mean(pxx_raw_log[freq_idx_noise_surrounding, :])
        ratio_noise_raw = 10 ** ((mean_pxx_raw_noise - mean_pxx_raw_noise_surrounding) / 10)

        mean_pxx_clean_noise = np.mean(pxx_clean_log[freq_idx_noise, :])
        mean_pxx_clean_noise_surrounding = np.mean(pxx_clean_log[freq_idx_noise_surrounding, :])
        ratio_noise_clean = 10 ** ((mean_pxx_clean_noise - mean_pxx_clean_noise_surrounding) / 10)

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

    def adaptive_cleaning(self, clean_data, raw_data, noise_freq,zapline_config,cleaning_too_strong_once):
        """
        Adjust the cleaning process if it was too weak or too strong.
        """
        from scipy.signal import windows

        # Compute the PSD of the clean data
        nperseg = int(self.config['winSizeCompleteSpectrum'] * self.sampling_rate)
        noverlap = int(0.5 * nperseg)  # 50% overlap

        f, pxx_clean = signal.welch(
            clean_data,
            fs=self.sampling_rate,
            window=windows.hann(nperseg, sym=True),
            nperseg=nperseg,
            noverlap=noverlap,
            axis=0
        )
        pxx_clean_log = 10 * np.log10(pxx_clean)

        # Determine center power by checking lower and upper third around noise frequency
        detectionWinsize = zapline_config['detectionWinsize']
        freq_range = (f > noise_freq - (detectionWinsize / 2)) & (f < noise_freq + (detectionWinsize / 2))
        this_fine_data = np.mean(pxx_clean_log[freq_range, :], axis=1)

        # Calculate thirds of the data
        third = int(np.round(len(this_fine_data) / 3))
        if third == 0:
            third = 1  # Ensure at least one sample

        indices = np.concatenate((np.arange(0, third), np.arange(2 * third, len(this_fine_data))))
        center_this_data = np.mean(this_fine_data[indices])

        # Measure of variation using lower quantile
        mean_lower_quantile = np.mean([
            np.quantile(this_fine_data[0:third], 0.05),
            np.quantile(this_fine_data[2 * third:], 0.05)
        ])

        # Compute thresholds
        freq_detect_mult_fine = zapline_config['freqDetectMultFine']
        remaining_noise_thresh_upper = center_this_data + freq_detect_mult_fine * (center_this_data - mean_lower_quantile)
        remaining_noise_thresh_lower = center_this_data - freq_detect_mult_fine * (center_this_data - mean_lower_quantile)
        zapline_config['remaining_noise_thresh_upper'] = remaining_noise_thresh_upper
        zapline_config['remaining_noise_thresh_lower'] = remaining_noise_thresh_lower
        
        # Frequency indices for upper and lower checks
        freq_idx_upper_check = (f > noise_freq + zapline_config['detailedFreqBoundsUpper'][0]) & \
                            (f < noise_freq + zapline_config['detailedFreqBoundsUpper'][1])
        freq_idx_lower_check = (f > noise_freq + zapline_config['detailedFreqBoundsLower'][0]) & \
                            (f < noise_freq + zapline_config['detailedFreqBoundsLower'][1])
                            
        # Store frequency indices for plotting
        zapline_config['thisFreqidxUppercheck'] = freq_idx_upper_check
        zapline_config['thisFreqidxLowercheck'] = freq_idx_lower_check
        
        # Proportions for cleaning assessment
        numerator_upper = float(np.sum(
            np.mean(pxx_clean_log[freq_idx_upper_check, :], axis=1) > remaining_noise_thresh_upper
        ))
        denominator_upper = float(np.sum(freq_idx_upper_check)) or 1.0
        proportion_above_upper = numerator_upper / denominator_upper
        cleaning_too_weak = proportion_above_upper > zapline_config['maxProportionAboveUpper']
        zapline_config['proportion_above_upper'] = proportion_above_upper
        
        if cleaning_too_weak:
            print("Cleaning too weak! ")
        numerator_lower = float(np.sum(
            np.mean(pxx_clean_log[freq_idx_lower_check, :], axis=1) < remaining_noise_thresh_lower
        ))
        denominator_lower = float(np.sum(freq_idx_lower_check)) or 1.0
        proportion_below_lower = numerator_lower / denominator_lower
        cleaning_too_strong = proportion_below_lower > zapline_config['maxProportionBelowLower']
        zapline_config['proportion_below_lower'] = proportion_below_lower

        # Adjust cleaning parameters based on the assessment
        cleaning_done = True
        if zapline_config['adaptiveNremove'] and zapline_config['adaptiveSigma']:
            if cleaning_too_strong and zapline_config['noiseCompDetectSigma'] < zapline_config['maxsigma']:
                cleaning_too_strong_once = True
                zapline_config['noiseCompDetectSigma'] = min(
                    zapline_config['noiseCompDetectSigma'] + 0.25,
                    zapline_config['maxsigma']
                )
                cleaning_done = False
                # Decrease current minimum components to remove, but never below the original baseline
                base_min = int(zapline_config.get('baseFixedNremove', zapline_config.get('fixedNremove', 1)))
                zapline_config['fixedNremove'] = max(int(zapline_config['fixedNremove']) - 1, base_min)
                print(f"Cleaning too strong! Increasing sigma for noise component detection to {zapline_config['noiseCompDetectSigma']} "
                    f"and setting minimum number of removed components to {zapline_config['fixedNremove']}.")
                return cleaning_done, zapline_config, cleaning_too_strong_once

            elif cleaning_too_weak and not cleaning_too_strong_once and zapline_config['noiseCompDetectSigma'] > zapline_config['minsigma']:
                zapline_config['noiseCompDetectSigma'] = max(
                    zapline_config['noiseCompDetectSigma'] - 0.25,
                    zapline_config['minsigma']
                )
                cleaning_done = False
                zapline_config['fixedNremove'] += 1
                print(f"Cleaning too weak! Reducing sigma for noise component detection to {zapline_config['noiseCompDetectSigma']} "
                    f"and setting minimum number of removed components to {zapline_config['fixedNremove']}.")

        return cleaning_done, zapline_config, cleaning_too_strong_once



    def nt_smooth(self, x, T, n_iterations=1, nodelayflag=False):
        """
        Smooth the data by convolution with a square window.

        Parameters:
        x (numpy.ndarray): The input data to smooth. Shape: (samples, channels) or (samples, channels, ...)
        T (float): The window size (can be fractional).
        n_iterations (int): Number of iterations of smoothing (default is 1).
        nodelayflag (bool): If True, compensate for delay introduced by filtering.

        Returns:
        numpy.ndarray: Smoothed data with the same shape as input.
        """
        from scipy.signal import lfilter

        # Ensure x is at least 2D
        if x.ndim < 2:
            x = x[:, np.newaxis]

        # Split T into integer and fractional parts
        integ = int(np.floor(T))
        frac = T - integ

        # If the window size exceeds data length, replace data with mean
        if integ >= x.shape[0]:
            x = np.tile(np.mean(x, axis=0), (x.shape[0], 1))
            return x

        # Remove onset step (similar to MATLAB code)
        mn = np.mean(x[:integ + 1, :], axis=0)
        x = x - mn

        if n_iterations == 1 and frac == 0:
            # Faster convolution using cumulative sum (similar to MATLAB)
            cumsum = np.cumsum(x, axis=0)
            x[integ:, :] = (cumsum[integ:, :] - cumsum[:-integ, :]) / T
        else:
            # Construct the initial filter kernel B
            B = np.concatenate((np.ones(integ), [frac])) / T

            # Iteratively convolve B with [ones(integ), frac] / T for n_iterations-1 times
            for _ in range(n_iterations - 1):
                B = np.convolve(B, np.concatenate((np.ones(integ), [frac]))) / T

            # Apply the filter using causal filtering (similar to MATLAB's filter)
            # For multi-dimensional data, apply filter along the first axis (samples)
            for channel in range(x.shape[1]):
                x[:, channel] = lfilter(B, 1, x[:, channel])

        # Restore the mean value that was subtracted earlier
        x = x + mn

        # Delay compensation if nodelayflag is set to True
        if nodelayflag:
            shift = int(round(T / 2 * n_iterations))
            if shift > 0:
                # Shift the data forward and pad the end with zeros
                padding = np.zeros((shift, x.shape[1]))
                x = np.vstack((x[shift:, :], padding))
            else:
                pass  # No shift needed

        return x




    def nt_pca(self, x, shifts=None, nkeep=None, threshold=0, w=None):
        """
        Apply PCA with time shifts and retain a specified number of components.

        Parameters:
        - x: data matrix (n_samples, n_channels) or list of arrays for cell-like data
        - shifts: array of shifts to apply (default: [0])
        - nkeep: number of components to keep (default: all)
        - threshold: discard PCs with eigenvalues below this (default: 0)
        - w: weights (optional)
            - If x is numeric: w can be 1D (n_samples,) or 2D (n_samples, n_channels)
            - If x is a list: w should be a list of arrays matching x's structure

        Returns:
        - z: principal components
            - If x is numeric: numpy.ndarray of shape (numel(idx), PCs, trials)
            - If x is a list: list of numpy.ndarrays, each of shape (numel(idx), PCs)
        - idx: indices of x that map to z
        """

        # Ensure shifts is a numpy array
        if shifts is None:
            shifts = np.array([0])
        else:
            shifts = np.array(shifts).flatten()
            if len(shifts) == 0:
                shifts = np.array([0])
        if np.any(shifts < 0):
            raise ValueError("All shifts must be non-negative.")

        # Adjust shifts to make them non-negative
        min_shift: int = int(np.min(shifts))
        offset = max(0, -min_shift)
        shifts = shifts + offset
        idx = offset + np.arange(x.shape[0] - max(shifts))  # x[idx] maps to z
        # Determine if x is numeric or list (cell-like)
        if isinstance(x, list):
            o = len(x)
            if o == 0:
                raise ValueError("Input list 'x' is empty.")
            m, n = x[0].shape
            if w is not None and not isinstance(w, list):
                raise ValueError("Weights 'w' must be a list when 'x' is a list.")
            tw = 0
            # Compute covariance
            c, tw = self.nt_cov(x, shifts, w)
        elif isinstance(x, np.ndarray):
            if x.ndim not in [2, 3]:
                raise ValueError("Input 'x' must be a 2D or 3D numpy.ndarray or a list of 2D arrays.")
            m, n = x.shape[:2]
            o = x.shape[2] if x.ndim == 3 else 1
            c, tw = self.nt_cov(x, shifts, w)
        else:
            raise TypeError("Input 'x' must be a numpy.ndarray or a list of numpy.ndarrays.")

        # Perform PCA
        topcs, evs = self.nt_pcarot(c, nkeep, threshold)

        # Apply PCA matrix to time-shifted data
        if isinstance(x, list):
            z = []
            for k in range(o):
                shifted = self.nt_multishift(x[k], shifts)  # Shape: (numel(idx), n * nshifts)
                # Project onto PCA components
                z_k = np.dot(shifted, topcs)
                z.append(z_k)
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                shifted = self.nt_multishift(x, shifts)  # Shape: (numel(idx), n * nshifts)
                z = shifted @ topcs # Shape: (numel(idx), PCs)
            elif x.ndim == 3:
                z = np.zeros((len(idx), topcs.shape[1], o))
                for k in range(o):
                    shifted = self.nt_multishift(x[:, :, k], shifts)  # Shape: (numel(idx), n * nshifts)
                    shifted = shifted.reshape(-1, 1)  # Shape becomes (18000, 1)
                    z[:, :, k] = np.dot(shifted, topcs)  # Shape: (numel(idx), PCs)
        else:
            # This case should have been handled earlier
            z = None

        return z, idx



    def nt_multishift(self,x, shifts):
        """
        Apply multiple shifts to a matrix.

        Parameters:
        x (numpy.ndarray): Input data to shift. Shape can be 1D, 2D, or 3D.
                            - 1D: (samples,)
                            - 2D: (samples, channels)
                            - 3D: (samples, channels, trials)
        shifts (array-like): Array of non-negative integer shifts.

        Returns:
        numpy.ndarray: Shifted data with increased channel dimension.
                    - 1D input becomes 2D: (samples_shifted, shifts.size)
                    - 2D input becomes 3D: (samples_shifted, channels * shifts.size, trials)
        """
        x = np.asarray(x)
        shifts = np.asarray(shifts).flatten()
        nshifts = shifts.size

        # Input validation
        if np.any(shifts < 0):
            raise ValueError('Shifts should be non-negative')
        if x.shape[0] < np.max(shifts):
            raise ValueError('Shifts should be no larger than the number of time samples in x')

        # Handle different input dimensions by expanding to 3D
        if x.ndim == 1:
            x = x[:, np.newaxis, np.newaxis]  # (samples, 1, 1)
        elif x.ndim == 2:
            x = x[:, :, np.newaxis]  # (samples, channels, 1)
        elif x.ndim > 3:
            raise ValueError('Input data has more than 3 dimensions, which is not supported.')

        m, n, o = x.shape  # samples x channels x trials

        # If only one shift and it's zero, return the original data
        if nshifts == 1 and shifts[0] == 0:
            return x.squeeze(axis=-1)

        max_shift: int = int(np.max(shifts))
        N: int = m - max_shift  # Number of samples after shifting

        # Initialize output array
        z = np.empty((N, n * nshifts, o), dtype=x.dtype)

        for trial in range(o):
            for channel in range(n):
                y = x[:, channel, trial]  # (samples,)
                for s_idx, shift in enumerate(shifts):
                    if shift == 0:
                        shifted_y = y[:N]
                    else:
                        shifted_y = y[shift:shift + N]
                    # Place the shifted data in the correct position
                    z[:, channel * nshifts + s_idx, trial] = shifted_y

        return z.squeeze(axis=-1)





    def nt_cov(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        shifts: Optional[Union[List[int], np.ndarray]] = None,
        w: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate time-shifted covariance of the data.

        Parameters:
            x (Union[np.ndarray, List[np.ndarray]]): Data matrix or list of matrices.
                - If numeric:
                    - 1D: (n_samples,)
                    - 2D: (n_samples, n_channels)
                    - 3D: (n_samples, n_channels, n_trials)
                - If list: list of 2D numpy arrays (cell array equivalent).
            shifts (Union[List[int], np.ndarray]): Array-like of non-negative integer shifts.
            w (Optional[Union[np.ndarray, List[np.ndarray]]]): Weights (optional).
                - If numeric:
                    - 1D array for 1D or 2D `x`.
                    - 2D array for 3D `x`.
                - If list: list of weight matrices corresponding to each cell.

        Returns:
            Tuple[np.ndarray, float]: Covariance matrix and total weight.
                - c: covariance matrix (numpy.ndarray) with shape (n_channels * nshifts, n_channels * nshifts).
                - tw: total weight (float).
        """
        # Convert shifts to a NumPy array and flatten to 1D
        if shifts is None:
            shifts = np.array([0])
        else:
            shifts = np.asarray(shifts).flatten()
        shifts = cast(np.ndarray, shifts)
        if np.any(shifts < 0):
            raise ValueError("Shifts must be non-negative integers.")
        nshifts = len(shifts)

        # Validate input data is not empty
        if x is None or (isinstance(x, np.ndarray) and x.size == 0) or (isinstance(x, list) and len(x) == 0):
            raise ValueError("Input data `x` is empty.")

        # Initialize covariance matrix and total weight
        c = None
        tw = 0.0

        # Determine if input is a list (cell array) or numpy array
        if isinstance(x, list):
            # Handle list input (cell array equivalent)
            if w is not None and not isinstance(w, list):
                raise ValueError("Weights `w` must be a list if `x` is a list (cell array).")
            
            # Number of cells/trials not required explicitly here
            # Determine number of channels
            if len(x) == 0:
                raise ValueError("Input list `x` is empty.")
            first_shape = x[0].shape
            if first_shape == ():
                n_channels = 1
            elif x[0].ndim == 1:
                n_channels = 1
            elif x[0].ndim == 2:
                n_channels = x[0].shape[1]
            else:
                raise ValueError(f"Data in cell {0} has unsupported number of dimensions: {x[0].ndim}")

            c = np.zeros((n_channels * nshifts, n_channels * nshifts), dtype=np.float64)

            for idx, data in enumerate(x):
                # Validate data dimensions
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"Element {idx} of input list `x` is not a numpy.ndarray.")
                if data.ndim != 2:
                    raise ValueError(f"Data in cell {idx} must be 2D, got {data.ndim}D.")
                n_samples, n_channels_current = data.shape
                if n_channels_current != n_channels:
                    raise ValueError(f"All cells must have the same number of channels. Cell {idx} has {n_channels_current} channels, expected {n_channels}.")

                # Handle weights
                if w is not None:
                    weight = w[idx]
                    if not isinstance(weight, np.ndarray):
                        raise TypeError(f"Weight for cell {idx} must be a numpy.ndarray.")
                    if weight.size == 0:
                        raise ValueError(f"Weight for cell {idx} is empty.")
                    if weight.ndim == 1:
                        weight = weight[:, np.newaxis]  # Shape: (n_samples, 1)
                    elif weight.ndim == 2 and weight.shape[1] == 1:
                        pass  # Shape is already (n_samples, 1)
                    else:
                        raise ValueError(f"Weight for cell {idx} must be 1D or 2D with a single column. Got shape {weight.shape}.")
                # Apply shifts
                if not np.all(shifts == 0):
                    xx = self.nt_multishift(data, shifts)  # Shape: (n_samples * nshifts, n_channels)
                    if w is not None:
                        ww = self.nt_multishift(weight, shifts)  # Shape: (n_samples * nshifts, 1)
                        # Take the minimum weight across shifts for each sample
                        ww_min = np.min(ww, axis=1, keepdims=True)  # Shape: (n_samples * nshifts, 1)
                    else:
                        ww_min = np.ones((xx.shape[0], 1), dtype=np.float64)
                else:
                    xx = data.copy()  # Shape: (n_samples, n_channels)
                    if w is not None:
                        ww_min = weight.copy()  # Shape: (n_samples, 1)
                    else:
                        ww_min = np.ones((xx.shape[0], 1), dtype=np.float64)

                # Multiply each row by its corresponding weight
                if w is not None:
                    xx = self.nt_vecmult(xx, ww_min)  # Shape: (time_shifted x channels)
                
                # Accumulate covariance
                c += np.dot(xx.T, xx)  # Shape: (n_channels * nshifts, n_channels * nshifts)

                # Accumulate total weight
                if w is not None:
                    if not np.all(shifts == 0):
                        tw += np.sum(ww_min)
                    else:
                        tw += np.sum(weight)
                else:
                    tw += xx.shape[0]  # Number of samples

        elif isinstance(x, np.ndarray):
            # Handle NumPy array input
            data = x.copy()
            # original_shape not used; keep minimal state

            # Determine data dimensionality
            if data.ndim == 1:
                data = data[:, np.newaxis]  # Shape: (n_samples, 1)
                n_channels = 1
                n_trials = 1
            elif data.ndim == 2:
                n_channels = data.shape[1]
                n_trials = 1
            elif data.ndim == 3:
                n_channels = data.shape[1]
                n_trials = data.shape[2]
            else:
                raise ValueError(f"Input data has unsupported number of dimensions: {data.ndim}")

            # Preallocate covariance matrix
            c = np.zeros((n_channels * nshifts, n_channels * nshifts), dtype=np.float64)

            # Iterate over trials
            for trial in range(n_trials):
                if data.ndim == 3:
                    trial_data = data[:, :, trial]  # Shape: (n_samples, n_channels)
                    if w is not None:
                        if not isinstance(w, np.ndarray):
                            raise TypeError("Weights `w` must be a numpy.ndarray when `x` is a numpy.ndarray.")
                        trial_weight = w[:, trial]  # Shape: (n_samples,)
                else:
                    trial_data = data.copy()  # Shape: (n_samples, n_channels)
                    if w is not None:
                        if not isinstance(w, np.ndarray):
                            raise TypeError("Weights `w` must be a numpy.ndarray when `x` is a numpy.ndarray.")
                        trial_weight = w.copy()  # Shape: (n_samples, 1)

                # Apply shifts
                if not np.all(shifts == 0):
                    xx = self.nt_multishift(trial_data, shifts)  # Shape: (n_samples * nshifts, n_channels)
                    if w is not None:
                        ww = self.nt_multishift(trial_weight, shifts)  # Shape: (n_samples * nshifts, 1)
                        # Take the minimum weight across shifts for each sample
                        ww_min = np.min(ww, axis=1, keepdims=True)  # Shape: (n_samples * nshifts, 1)
                    else:
                        ww_min = np.ones((xx.shape[0], 1), dtype=np.float64)
                else:
                    xx = trial_data.copy()  # Shape: (n_samples, n_channels)
                    if w is not None:
                        ww_min = trial_weight.copy()  # Shape: (n_samples, 1)
                    else:
                        ww_min = np.ones((xx.shape[0], 1), dtype=np.float64)

                # Multiply each row by its corresponding weight
                if w is not None:
                    xx = self.nt_vecmult(xx, ww_min)  # Shape: (time_shifted x channels)

                # Accumulate covariance
                c += np.dot(xx.T, xx)  # Shape: (n_channels * nshifts, n_channels * nshifts)

                # Accumulate total weight
                if w is not None:
                    if not np.all(shifts == 0):
                        tw += np.sum(ww_min)
                    else:
                        tw += np.sum(trial_weight)
                else:
                    tw += xx.shape[0]  # Number of samples

        else:
            raise TypeError("Input `x` must be a numpy.ndarray or a list of numpy.ndarray (cell array).")

        return c, tw


    def nt_vecmult(self, xx: np.ndarray, ww: np.ndarray) -> np.ndarray:
        """
        Multiply each row of 'xx' by the corresponding weight in 'ww'.

        Parameters:
            xx (np.ndarray): Data array (time_shifted x channels).
            ww (np.ndarray): Weights array (time_shifted x 1).

        Returns:
            weighted_xx (np.ndarray): Weighted data (time_shifted x channels).
        """
        # Ensure that 'ww' is a column vector
        if ww.ndim == 1:
            ww = ww[:, np.newaxis]
        elif ww.ndim == 2 and ww.shape[1] != 1:
            raise ValueError("Weights 'ww' must have a single column or be a 1D array.")
        
        # Element-wise multiplication with broadcasting
        weighted_xx = xx * ww  # Shape: (time_shifted x channels)
        return weighted_xx
    def nt_pcarot(self, cov, nkeep=None, threshold=None, N=None):
        """
        Calculate PCA rotation matrix from covariance matrix.

        Parameters:
        - cov (numpy.ndarray): Covariance matrix (symmetric, positive semi-definite).
        - nkeep (int, optional): Number of principal components to keep.
        - threshold (float, optional): Discard components with eigenvalues below this fraction of the largest eigenvalue.
        - N (int, optional): Number of top eigenvalues and eigenvectors to compute.

        Returns:
        - topcs (numpy.ndarray): PCA rotation matrix (eigenvectors), shape (n_features, n_components).
        - eigenvalues (numpy.ndarray): PCA eigenvalues, shape (n_components,).
        """
        from scipy.sparse.linalg import eigsh
        from scipy.linalg import eigh
        
        # Validate covariance matrix
        if not isinstance(cov, np.ndarray):
            raise TypeError("Covariance matrix 'cov' must be a numpy.ndarray.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance matrix 'cov' must be a square (2D) array.")

        n_features = cov.shape[0]

        # Compute eigenvalues and eigenvectors
        if N is not None:
            if not isinstance(N, int) or N <= 0:
                raise ValueError("'N' must be a positive integer.")
            N = min(N, n_features)
            eigenvalues_all, eigenvectors_all = eigsh(cov, k=N, which='LM')  # 'LM' selects largest magnitude eigenvalues
            # Ensure real parts
            eigenvalues_all = np.real(eigenvalues_all)
            eigenvectors_all = np.real(eigenvectors_all)
        else:
            eigenvalues_all, eigenvectors_all = eigh(cov)
            # Ensure real parts
            eigenvalues_all = np.real(eigenvalues_all)
            eigenvectors_all = np.real(eigenvectors_all)
            # Reverse to descending order
            eigenvalues_all = eigenvalues_all[::-1]
            eigenvectors_all = eigenvectors_all[:, ::-1]
            # Filter out negative eigenvalues
            eigenvalues_all=np.abs(eigenvalues_all)
            # positive_idx = eigenvalues_all > 0
            # eigenvalues_all = eigenvalues_all[positive_idx]
            # eigenvectors_all = eigenvectors_all[:, positive_idx]

        # Define a small tolerance to handle numerical precision issues (optional)


        # Select top N eigenvalues and eigenvectors if N is specified and not already done
        if N is None:
            eigenvalues = eigenvalues_all
            eigenvectors = eigenvectors_all
        else:
            eigenvalues = eigenvalues_all
            eigenvectors = eigenvectors_all

        # Apply threshold
        if threshold is not None:
            if eigenvalues[0] == 0:
                raise ValueError("The largest eigenvalue is zero; cannot apply threshold.")
            valid_indices = np.where(eigenvalues / eigenvalues[0] > threshold)[0]
            eigenvalues = eigenvalues[valid_indices]
            eigenvectors = eigenvectors[:, valid_indices]

        # Apply nkeep
        if nkeep is not None:
            if not isinstance(nkeep, int) or nkeep <= 0:
                raise ValueError("'nkeep' must be a positive integer.")
            nkeep = min(nkeep, eigenvectors.shape[1])
            eigenvalues = eigenvalues[:nkeep]
            eigenvectors = eigenvectors[:, :nkeep]

        topcs = eigenvectors
        return topcs, eigenvalues

    def nt_xcov(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        shifts: Optional[Union[List[int], np.ndarray]] = None,
        w: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the cross-covariance of x and time-shifted y.

        Parameters:
            x (Union[np.ndarray, List[np.ndarray]]): Data array x (time x channels x trials) or list of 2D arrays.
            y (Union[np.ndarray, List[np.ndarray]]): Data array y (time x channels x trials) or list of 2D arrays.
            shifts (Optional[Union[List[int], np.ndarray]]): Array of non-negative integer time shifts (default: [0]).
            w (Optional[Union[np.ndarray, List[np.ndarray]]]): Optional weights array (time x 1 x trials) or (time x channels x trials).

        Returns:
            Tuple[np.ndarray, float]: Cross-covariance matrix and total weight.
                - c: cross-covariance matrix.
                - tw: total weight.
        """
        # If numpy arrays are 2D, expand to 3D for consistent processing
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.ndim == 2 and y.ndim == 2:
                x = x[:, :, np.newaxis]
                y = y[:, :, np.newaxis]
                if w is not None:
                    if not isinstance(w, np.ndarray):
                        raise TypeError("Weights `w` must be a numpy.ndarray when `x`/`y` are numpy.ndarray types.")
                    if w.ndim == 2:
                        w = w[:, :, np.newaxis]
        if shifts is None:
            shifts = np.array([0])
        else:
            shifts = np.asarray(shifts).flatten()
        shifts = cast(np.ndarray, shifts)

        if np.any(shifts < 0):
            raise ValueError('Shifts must be non-negative integers')

        # Validate dimensions
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                raise ValueError("Lists `x` and `y` must have the same length.")
            if w is not None and not isinstance(w, list):
                raise ValueError("Weights `w` must be a list if `x` and `y` are lists.")
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.ndim != y.ndim:
                raise ValueError("Arrays `x` and `y` must have the same number of dimensions.")
            if x.shape[0] != y.shape[0]:
                raise ValueError("`x` and `y` must have the same number of time samples.")
            if x.ndim > 2:
                if x.shape[2] != y.shape[2]:
                    raise ValueError("`x` and `y` must have the same number of trials.")
                if w is not None:
                    if not isinstance(w, np.ndarray):
                        raise TypeError("Weights `w` must be a numpy.ndarray when `x`/`y` are numpy.ndarray types.")
                    if x.shape[2] != w.shape[2]:
                        raise ValueError("`x` and `w` must have the same number of trials.")
        else:
            raise TypeError("`x` and `y` must both be either lists or numpy.ndarray types.")

        nshifts = shifts.size

        # Initialize covariance matrix and total weight
        c = None
        tw = 0.0

        # Handle list inputs (equivalent to cell arrays in MATLAB)
        if isinstance(x, list):
            if w is not None and not isinstance(w, list):
                raise ValueError("Weights `w` must be a list if `x` is a list (cell array).")

            o = len(x)  # Number of cells/trials
            if o == 0:
                raise ValueError("Input list `x` is empty.")
            
            # Determine number of channels from the first cell
            # shapes of first elements inferred below when needed
            if x[0].ndim == 1:
                n_channels_x = 1
            elif x[0].ndim == 2:
                n_channels_x = x[0].shape[1]
            else:
                raise ValueError(f"Data in cell 0 of `x` has unsupported number of dimensions: {x[0].ndim}")

            if y[0].ndim == 1:
                n_channels_y = 1
            elif y[0].ndim == 2:
                n_channels_y = y[0].shape[1]
            else:
                raise ValueError(f"Data in cell 0 of `y` has unsupported number of dimensions: {y[0].ndim}")

            # Initialize cross-covariance matrix
            c = np.zeros((n_channels_x, n_channels_y * nshifts), dtype=np.float64)

            for idx in range(o):
                data_x = x[idx]
                data_y = y[idx]

                # Validate data dimensions
                if not isinstance(data_x, np.ndarray) or not isinstance(data_y, np.ndarray):
                    raise TypeError(f"Elements in lists `x` and `y` must be numpy.ndarray types. Found types: {type(data_x)}, {type(data_y)} at index {idx}.")
                if data_x.ndim != 2 or data_y.ndim != 2:
                    raise ValueError(f"Data in lists `x` and `y` must be 2D arrays. Found dimensions: {data_x.ndim}, {data_y.ndim} at index {idx}.")

                if data_x.shape[1] != n_channels_x:
                    raise ValueError(f"All cells in `x` must have {n_channels_x} channels. Found {data_x.shape[1]} at index {idx}.")
                if data_y.shape[1] != n_channels_y:
                    raise ValueError(f"All cells in `y` must have {n_channels_y} channels. Found {data_y.shape[1]} at index {idx}.")

                # Handle weights
                if w is not None:
                    weight = w[idx]
                    if not isinstance(weight, np.ndarray):
                        raise TypeError(f"Weight for cell {idx} must be a numpy.ndarray.")
                    if weight.size == 0:
                        raise ValueError(f"Weight for cell {idx} is empty.")
                    if weight.ndim == 1:
                        weight = weight[:, np.newaxis]  # Shape: (n_samples, 1)
                    elif weight.ndim == 2 and weight.shape[1] == 1:
                        pass  # Shape is already (n_samples, 1)
                    else:
                        raise ValueError(f"Weight for cell {idx} must be 1D or 2D with a single column. Got shape {weight.shape}.")

                # Apply shifts to y
                y_shifted = self.nt_multishift(data_y, shifts)  # Shape: (n_samples_shifted, n_channels_y * nshifts)

                # Truncate x to match the shifted y's time dimension
                if not np.all(shifts == 0):
                    # Apply shifts to x if necessary (though MATLAB does not shift x in cross-covariance)
                    # Assuming x is not shifted, only y is
                    # Thus, truncate x to match the shifted y's number of samples
                    x_truncated = data_x[:y_shifted.shape[0], :]  # Shape: (n_samples_shifted, n_channels_x)
                else:
                    x_truncated = data_x.copy()

                # Handle weights: multiply x by w
                if w is not None:
                    # Unfold x and w, multiply, fold back
                    x_weighted = self.nt_fold(
                        self.nt_vecmult(
                            self.nt_unfold(x_truncated),
                            self.nt_unfold(weight[:y_shifted.shape[0], :]),
                        ),
                        data_x.shape[0],
                    )[: y_shifted.shape[0], :]  # Ensure matching time dimension
                    x_truncated = x_weighted  # Shape: (n_samples_shifted, n_channels_x)

                    # Update total weight
                    if not np.all(shifts == 0):
                        # Take minimum weight across shifts
                        ww = self.nt_multishift(weight[:y_shifted.shape[0], :], shifts)  # Shape: (n_samples_shifted * nshifts, 1)
                        ww_min = np.min(ww, axis=1, keepdims=True)
                        tw += np.sum(ww_min)
                    else:
                        tw += np.sum(weight[:y_shifted.shape[0], :])
                else:
                    tw += x_truncated.shape[0]  # Number of samples

                # Accumulate cross-covariance
                c += np.dot(x_truncated.T, y_shifted)  # Shape: (n_channels_x, n_channels_y * nshifts)

            # Handle NumPy array inputs
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # Validate dimensions
            if x.ndim != y.ndim:
                raise ValueError("Arrays `x` and `y` must have the same number of dimensions.")
            if x.shape[0] != y.shape[0]:
                raise ValueError("`x` and `y` must have the same number of time samples.")
            if x.ndim > 2:
                if x.shape[2] != y.shape[2]:
                    raise ValueError("`x` and `y` must have the same number of trials.")
                if w is not None:
                    if not isinstance(w, np.ndarray):
                        raise TypeError("Weights `w` must be a numpy.ndarray if `x` and `y` are numpy.ndarray types.")
                    if w.shape[0] != x.shape[0] or w.shape[2] != x.shape[2]:
                        raise ValueError("`w` must have the same number of time samples and trials as `x`.")

            mx, nx, ox = x.shape
            my, ny, oy = y.shape

            # Determine number of channels
            n_channels_x = nx
            n_channels_y = ny

            # Initialize cross-covariance matrix
            c = np.zeros((n_channels_x, n_channels_y * nshifts), dtype=np.float64)

            for trial in range(ox):
                data_x = x[:, :, trial]  # Shape: (n_samples, n_channels_x)
                data_y = y[:, :, trial]  # Shape: (n_samples, n_channels_y)

                # Apply shifts to y
                y_shifted= self.nt_multishift(data_y, shifts)  # Shape: (n_samples_shifted, n_channels_y * nshifts)

                # Truncate x to match the shifted y's time dimension
                if not np.all(shifts == 0):
                    # Assuming y is shifted, x is not; truncate x accordingly
                    x_truncated = data_x[:y_shifted.shape[0], :]  # Shape: (n_samples_shifted, n_channels_x)
                else:
                    x_truncated = data_x.copy()

                if w is not None:
                    # Extract weights for this trial and truncate
                    if not isinstance(w, np.ndarray):
                        raise TypeError("Weights `w` must be a numpy.ndarray when `x`/`y` are numpy.ndarray types.")
                    trial_weight = w[:, :, trial]  # Shape: (n_samples, channels or 1)
                    # if w.ndim == 2:
                    #     # For 3D `x`, weights are 2D (time x trials)
                    #     trial_weight = trial_weight  # Shape: (n_samples, trials)
                    # elif w.ndim == 1:
                    #     # For 1D or 2D `x`, weights are 1D
                    #     trial_weight = trial_weight[:, np.newaxis]  # Shape: (n_samples, 1)
                    # else:
                    #     raise ValueError(f"Unsupported weight dimensionality: {w.ndim}")

                    # Unfold x and w, multiply, fold back
                    x_weighted = self.nt_fold(
                        self.nt_vecmult(
                            self.nt_unfold(x_truncated),
                            self.nt_unfold(trial_weight[:y_shifted.shape[0], :]),
                        ),
                        mx,
                    )[: y_shifted.shape[0], :]  # Ensure matching time dimension

                    x_truncated = x_weighted  # Shape: (n_samples_shifted, n_channels_x)

                    if not np.all(shifts == 0):
                        # Take minimum weight across shifts
                        ww = self.nt_multishift(trial_weight[:y_shifted.shape[0], :], shifts)  # Shape: (n_samples_shifted * nshifts, 1)
                        ww_min = np.min(ww, axis=1, keepdims=True)
                        tw += np.sum(ww_min)
                    else:
                        tw += np.sum(trial_weight[:y_shifted.shape[0], :])
                else:
                    tw += x_truncated.shape[0]  # Number of samples

                # Accumulate cross-covariance
                c += np.dot(x_truncated.T, y_shifted)  # Shape: (n_channels_x, n_channels_y * nshifts)

        else:
            raise TypeError("`x` and `y` must both be either lists or numpy.ndarray types.")

        return c, tw


    def nt_bias_fft(self, x: np.ndarray, freq: np.ndarray, nfft: int) -> tuple:
        """
        Compute covariance matrices with and without filter bias using FFT.

        Parameters:
        - x (np.ndarray): Data matrix.
            - 2D: (n_samples, n_channels)
            - 3D: (n_samples, n_channels, n_trials)
        - freq (np.ndarray): Normalized frequencies to retain.
            - 1D array: Individual frequencies.
            - 2D array: Frequency bands with two rows (start and end frequencies).
        - nfft (int): FFT size.

        Returns:
        - c0 (np.ndarray): Unbiased covariance matrix.
        - c1 (np.ndarray): Biased covariance matrix after applying the frequency filter.
        """
        from scipy.fft import fft
        from scipy.signal import windows


        # Input Validation
        if np.max(freq) > 0.5:
            raise ValueError("Frequencies should be <= 0.5")
        if nfft > x.shape[0]:
            raise ValueError("nfft too large")

        # Initialize Filter
        filt: np.ndarray = np.zeros(nfft // 2 + 1)

        if freq.ndim == 1:
            for k in range(freq.shape[0]):
                idx = int(round(freq[k] * nfft + 0.5))
                if idx >= len(filt):
                    raise ValueError(f"Frequency index {idx} out of bounds for filter of length {len(filt)}.")
                filt[idx] = 1
        elif freq.shape[0] == 2:
            for k in range(freq.shape[1]):
                start_idx = int(round(freq[0, k] * nfft + 0.5))
                end_idx = int(round(freq[1, k] * nfft + 0.5)) + 1
                if start_idx >= len(filt) or end_idx > len(filt):
                    raise ValueError(f"Frequency slice [{start_idx}:{end_idx}] out of bounds for filter of length {len(filt)}.")
                filt[start_idx:end_idx] = 1
        else:
            raise ValueError("freq should have one or two rows")

        # Symmetrize the Filter
        filt_full = np.concatenate([filt, np.flip(filt[1:-1])])

        # Hann Window
        w = windows.hann(nfft, sym=False)

        # Handle 2D and 3D Data
        if x.ndim == 2:
            n_samples, n_channels = x.shape
            n_trials = 1
            x = x[:, :, np.newaxis]  # Convert to 3D for uniform processing
        elif x.ndim == 3:
            n_samples, n_channels, n_trials = x.shape
        else:
            raise ValueError("Input data `x` must be 2D or 3D.")

        # Compute c0: Unbiased Covariance Matrix
        if n_trials > 1:
            x_reshaped = x.reshape(n_samples, n_channels * n_trials)
        else:
            x_reshaped = x.reshape(n_samples, n_channels)
        c0,_= self.nt_cov(x_reshaped)

        # Ensure c0 is 2D
        if c0.ndim == 0:
            c0 = np.array([[c0]])
        elif c0.ndim == 1:
            c0 = np.atleast_2d(c0)

        # Initialize c1: Biased Covariance Matrix
        c1 = np.zeros_like(c0)

        # Calculate Number of Frames
        nframes = int(np.ceil((n_samples - nfft / 2) / (nfft / 2)))

        for trial in range(n_trials):
            for k in range(nframes):
                idx = int(k * nfft // 2)
                idx = min(idx, n_samples - nfft)
                z = x[idx:idx + nfft, :, trial]  # (nfft, n_channels)
                z = self.nt_vecmult(z,w)  # Apply Hann window
                Z = fft(z, axis=0)
                Z = self.nt_vecmult(Z, filt_full)  # Apply filter
                cov_matrix = np.real(np.dot(Z.conj().T, Z))  # (n_channels, n_channels)
                c1 += cov_matrix

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
            idx = np.where(evs1 > keep2)[0]  # Extract the first element from the tuple
            topcs1 = topcs1[:, idx]
            evs1 = evs1[idx]

        # Apply PCA and whitening to the biased covariance
        evs1_sqrt=1.0 / np.sqrt(evs1)
        N = np.diag(evs1_sqrt)
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
    def nt_tsr(self, x, ref, shifts=None, wx=None, wref=None, keep=None, thresh=1e-20):
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
            
        # x = np.atleast_3d(x)  # Shape: (time, channels, trials) or (time, channels, 1)
        # ref = np.atleast_3d(ref)
        
        if wx is not None and wx.ndim == 2:
            wx = wx[:, np.newaxis, :]
            wx = np.atleast_3d(wx)
        if wref is not None and wref.ndim == 2:
            wref = wref[:, np.newaxis, :]
            wref = np.atleast_3d(wref)
        
        # Ensure x and ref are at least 3D

        # Check argument values for sanity
        if x.shape[0] != ref.shape[0]:
            raise ValueError('x and ref should have the same number of time samples')
        if x.ndim>=3:
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
        x = x[idx, :]  # Truncate x
        if wx is not None:
            wx = wx[idx, :]
        shifts = shifts + offset1  # Shifts are now non-negative
        
        # Adjust size of x
        offset2 = max(0, max(shifts))
        idx_ref = np.arange(0, ref.shape[0] - offset2)
        x = x[:len(idx_ref), :]  # Part of x that overlaps with time-shifted refs
        if wx is not None:
            wx = wx[:len(idx_ref), :]
        if x.ndim == 3:
            mx, nx, ox = x.shape
            mref, nref, oref = ref.shape
        elif x.ndim == 2:
            mx, nx = x.shape
            mref, nref = ref.shape
        else:
            raise ValueError('x should be 2D or 3D')
       
        
        # Consolidate weights into a single weight matrix
        w = np.zeros((mx, 1))
        if wx is None and wref is None:
            w[:] = 1
        elif wref is None:
            w = wx
        elif wx is None:
            wr = wref[:, :]
            wr_shifted = self.nt_multishift(wr, shifts)
            w[:, :] = np.min(wr_shifted, axis=1, keepdims=True)
        else:

            wr = wref[:, :]
            wr_shifted = self.nt_multishift(wr, shifts)
            w_min = np.min(wr_shifted, axis=1, keepdims=True)
            w[:, :] = np.minimum(w_min, wx[:w_min.shape[0], :])
        wx = w
        wref = np.zeros((mref, 1))
        wref[idx, :] = w
        
        # Remove weighted means
        x_demeaned, _ = self.nt_demean(x, wx)
        ref_demeaned, _ = self.nt_demean(ref, wref)
        
        # Equalize power of ref channels, then equalize power of ref PCs
        ref_normalized, _ = self.nt_normcol(ref_demeaned, wref)
        ref_pca, _ = self.nt_pca(ref_normalized, threshold=1e-6)
        ref_normalized_pca, _ = self.nt_normcol(ref_pca, wref)
        ref = ref_normalized_pca
        
        # Covariances and cross-covariance with time-shifted refs
        cref, twcref = self.nt_cov(ref, shifts, wref)
        cxref, twcxref = self.nt_xcov(x, ref, shifts, wx)
        
        # Regression matrix of x on time-shifted refs
        r = self.nt_regcov(cxref / twcxref, cref / twcref, keep=keep, threshold=thresh)
        
        # Clean x by removing regression on time-shifted refs
        y = np.zeros_like(x)

        ref_shifted = self.nt_multishift(ref[:, :], shifts)
        z = ref_shifted @ r
        y = x[:z.shape[0], :] - z
        
        y_demeaned, _ = self.nt_demean(y, wx)  # Multishift(ref) is not necessarily zero mean
        
        # idx for alignment
        idx_output = np.arange(offset1, offset1 + y.shape[0])
        w = wref
        
        # Return outputs
        return y_demeaned, idx_output, w

    def nt_mmat(self, x, m):
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

        # If m is 2D, perform simple matrix multiplication using
        if m.ndim == 2:
            y = self.nt_mmat0(x, m)

            # Ensure y has the correct shape by removing any singleton dimensions
            if y.ndim == 3 and y.shape[2] == 1:
                y = y.squeeze(-1)
            elif y.ndim == 2:
                pass  # Already correct
            else:
                # Handle unexpected dimensions
                y = np.squeeze(y)
            
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

    def nt_mmat0(self,x, m):
        """
        Performs matrix multiplication after unfolding x, then folds the result back.

        Parameters:
            x (np.ndarray): Input data. Can be 2D or 3D.
            m (np.ndarray): Matrix to multiply with. Should be 2D.

        Returns:
            y (np.ndarray): Result after multiplication and folding.
        """
        unfolded_x = self.nt_unfold(x)
        multiplied = unfolded_x @ m
        epochsize = x.shape[0]   # Assuming epochsize is the first dimension
        folded_y = self.nt_fold(multiplied, epochsize)
        return folded_y

    def nt_unfold(self,x):
        """
        Converts a 3D matrix (time x channel x trial) into a 2D matrix (time*trial x channel).

        Parameters:
            x (np.ndarray): Input data. Can be 2D or 3D.

        Returns:
            y (np.ndarray): Unfolded data.
        """
        if x.size == 0:
            return np.array([])
        else:
            if x.ndim == 3:
                m, n, p = x.shape
                if p > 1:
                    y = np.reshape(np.transpose(x, (0, 2, 1)), (m * p, n))
                else:
                    y = x
            else:
                y = x
        return y
    def nt_fold(self,x, epochsize):
        """
        Converts a 2D matrix (time*trial x channel) back into a 3D matrix (time x channel x trial).

        Parameters:
            x (np.ndarray): Input data. Should be 2D.
            epochsize (int): Number of samples per trial.

        Returns:
            y (np.ndarray): Folded data.
        """
        if x.size == 0:
            return np.array([])
        else:
            if x.shape[0] / epochsize > 1:
                trials = int(x.shape[0] / epochsize)
                y = np.transpose(np.reshape(x, (epochsize, trials, x.shape[1])), (0, 2, 1))
            else:
                y = x
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
        if x.ndim == 2:
            n_dim=2
            x=x[:,:,np.newaxis]
            w=w[:,:,np.newaxis]
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
        if n_dim == 2:
            x_demeaned = x_demeaned.squeeze(-1)  # Remove the last dimension if singleton
            mn = mn.squeeze(-1)
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
        topcs,eigenvalues = self.nt_pcarot(cyy)


        # Discard negligible regressor PCs
        if keep is not None:
            keep = max(keep, topcs.shape[1])
            topcs = topcs[:, :keep]
            eigenvalues = eigenvalues[:keep]

        # if threshold is not None and threshold > 0:
        #     idx_thresh = np.where(eigenvalues / np.max(eigenvalues) > threshold)[0]
        #     topcs = topcs[idx_thresh]
        #     eigenvalues = eigenvalues[idx_thresh]
        topcs, eigenvalues= self.normalize_topcs(eigenvalues, topcs, threshold)

        # Cross-covariance between data and regressor PCs
        cxy = cxy.T  # Transpose cxy to match dimensions
        r = topcs.T @ cxy

        # Projection matrix from regressor PCs
        r = self.nt_vecmult(r,1 / eigenvalues)

        # Projection matrix from regressors
        r = topcs @ r

        return r
    def normalize_topcs(self, eigenvalues, topcs, threshold=None):
        """
        Normalize and select top principal components based on a threshold.

        Parameters:
            eigenvalues (np.ndarray): 1D array of eigenvalues.
            topcs (np.ndarray): 2D array of top principal components (channels x PCs).
            threshold (float, optional): Threshold value for selecting eigenvalues.

        Returns:
            topcs_normalized (np.ndarray): Normalized top principal components.
            eigenvalues_selected (np.ndarray): Selected eigenvalues after thresholding.
        """
        if threshold is not None and threshold > 0:
            # Ensure eigenvalues is 1D
            if eigenvalues.ndim > 1:
                eigenvalues = np.diag(eigenvalues)
            
            # Compute the ratio and create a boolean mask
            ratio = eigenvalues / np.max(eigenvalues)
            mask = ratio > threshold  # Boolean array
            
            # Debug statements (optional)
            # print(f"Eigenvalues: {eigenvalues}")
            # print(f"Max Eigenvalue: {np.max(eigenvalues)}")
            # print(f"Threshold: {threshold}")
            # print(f"Ratio: {ratio}")
            # print(f"Mask: {mask}")
            
            # Select indices where the condition is True
            idx_thresh = np.where(mask)[0]
            
            # Debug statement (optional)
            # print(f"Selected indices: {idx_thresh}")
            
            # Select columns in topcs based on idx_thresh
            topcs_selected = topcs[:,idx_thresh]
            
            # Select corresponding eigenvalues
            eigenvalues_selected = eigenvalues[idx_thresh]
            
            return topcs_selected, eigenvalues_selected
        else:
            # If no threshold is provided, return the original arrays
            return topcs, eigenvalues
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
        threshold_old: float = float(np.max(data_vector))
        threshold: float = float(np.mean(data_vector) + sd_level * np.std(data_vector))
        n_remove: int = 0

        while threshold < threshold_old:
            flagged_points = data_vector > threshold
            data_vector = data_vector[~flagged_points]
            n_remove += np.sum(flagged_points)
            threshold_old = threshold
            threshold = float(np.mean(data_vector) + sd_level * np.std(data_vector))

        return n_remove, threshold
    def generate_output_figures(self, data, clean_data, noise_freq, zapline_config, pxx_raw_log, pxx_clean_log, pxx_removed_log, f, analytics, NremoveFinal):
        """
        Generate figures to visualize the results, replicating the MATLAB figures with the same colors.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib.gridspec as gridspec
        import os

        # Colors
        red = np.array([230, 100, 50]) / 256
        green = np.array([0, 97, 100]) / 256
        grey = np.array([0.2, 0.2, 0.2])

        # Prepare chunk indices for plotting
        chunk_indices = zapline_config.get('chunkIndices', None)
        if chunk_indices is None:
            print("Error: 'chunkIndices' not provided in 'zapline_config'.")
            return
        chunk_indices = np.array(chunk_indices)
        chunk_indices_plot = chunk_indices / self.sampling_rate / 60  # Convert to minutes

        # Compute chunk_indices_plot_individual
        chunk_indices_plot_individual = np.array([
            np.mean([chunk_indices_plot[i], chunk_indices_plot[i+1]]) for i in range(len(chunk_indices_plot)-1)
        ])

        # Frequency index for plotting around the noise frequency
        this_freq_idx_plot = (f >= noise_freq - 1.1) & (f <= noise_freq + 1.1)

        # Create figure and GridSpec
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(nrows=5, ncols=4, figure=fig, height_ratios=[1, 1, 1, 1, 1])

        # First row: ax1, ax4, ax5
        ax1 = fig.add_subplot(gs[0, 0])
        ax4 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[0, 2:4])

        # Second row: ax2 (span all columns)
        ax2 = fig.add_subplot(gs[1, :])

        # Third row: ax3 (span all columns)
        ax3 = fig.add_subplot(gs[2, :])

        # Fourth row: ax6 and ax7
        ax6 = fig.add_subplot(gs[3, 0:2])
        ax7 = fig.add_subplot(gs[3, 2:4])

        # Fifth row: ax8 (span all columns)
        ax8 = fig.add_subplot(gs[4, :])

        # Plot original power on ax1
        ax1.plot(f[this_freq_idx_plot], np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1), color=grey)
        ax1.set_xlim((float(f[this_freq_idx_plot][0] - 0.01), float(f[this_freq_idx_plot][-1])))

        # Y-axis limits
        remaining_noise_thresh_lower = zapline_config.get('remaining_noise_thresh_lower', None)
        remaining_noise_thresh_upper = zapline_config.get('remaining_noise_thresh_upper', None)
        coarse_freq_detect_power_diff = zapline_config.get('coarseFreqDetectPowerDiff', None)
        if remaining_noise_thresh_lower is not None and remaining_noise_thresh_upper is not None and coarse_freq_detect_power_diff is not None:
            ylim_lower = remaining_noise_thresh_lower - 0.25 * (remaining_noise_thresh_upper - remaining_noise_thresh_lower)
            ylim_upper = np.min(np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1)) + coarse_freq_detect_power_diff * 2
            ax1.set_ylim((float(ylim_lower), float(ylim_upper)))
        else:
            y_min: float = float(np.min(np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1)))
            y_max: float = float(np.max(np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1)))
            ax1.set_ylim((float(y_min - 0.25 * (y_max - y_min)), float(y_max + 0.25 * (y_max - y_min))))

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(labelsize=12)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power [10*log10 μV^2/Hz]')

        # Automatic frequency detection
        automatic_freq_detection = zapline_config.get('automaticFreqDetection', False)
        thresh = zapline_config.get('thresh', None)
        if automatic_freq_detection:
            if thresh is not None:
                ax1.plot(ax1.get_xlim(), [thresh[0], thresh[0]], color=red)
            ax1.set_title(f'Detected frequency: {noise_freq} Hz')
        else:
            ax1.set_title(f'Predefined frequency: {noise_freq} Hz')

        # Plot number of removed components on ax2
        ax2.cla()

        if NremoveFinal is None:
            print("Error: 'NremoveFinal' not provided.")
            return

        search_individual_noise = zapline_config.get('searchIndividualNoise', False)
        found_noise = zapline_config.get('foundNoise', [True]*len(NremoveFinal))
        nonoisehandle = None

        for i_chunk in range(len(chunk_indices_plot)-1):
            if (not search_individual_noise) or found_noise[i_chunk]:
                # Plot grey fill
                ax2.fill_between(
                    [chunk_indices_plot[i_chunk], chunk_indices_plot[i_chunk+1]],
                    0, NremoveFinal[i_chunk],
                    color=grey,
                    alpha=0.5
                )
            else:
                # Plot green fill
                nonoisehandle = ax2.fill_between(
                    [chunk_indices_plot[i_chunk], chunk_indices_plot[i_chunk+1]],
                    0, NremoveFinal[i_chunk],
                    color=green,
                    alpha=0.5
                )

        ax2.set_xlim((float(chunk_indices_plot[0]), float(chunk_indices_plot[-1])))
        ax2.set_ylim((0.0, float(max(NremoveFinal) + 1)))
        n_chunks = len(NremoveFinal)
        ax2.set_title(f'# removed comps in {n_chunks} chunks, μ = {round(np.mean(NremoveFinal), 2)}')
        ax2.tick_params(labelsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlabel('Time [minutes]')
        ax2.set_ylabel('Number of Components Removed')

        # Plot noise peaks on ax3
        noise_peaks = zapline_config.get('noisePeaks', None)
        if noise_peaks is None:
            print("Error: 'noisePeaks' not provided in 'zapline_config'.")
            return

        for i_chunk in range(len(chunk_indices_plot)-2):
            ax3.plot(
                [chunk_indices_plot[i_chunk+1], chunk_indices_plot[i_chunk+1]],
                [0, 1000],
                color=grey * 3
            )

        ax3.plot(chunk_indices_plot_individual, noise_peaks, color=grey)
        ax3.set_xlim((float(chunk_indices_plot[0]), float(chunk_indices_plot[-1])))
        max_diff = max([(max(noise_peaks)) - noise_freq, noise_freq - (min(noise_peaks))])
        if max_diff == 0:
            max_diff = 0.01
        ax3.set_ylim((float(noise_freq - max_diff * 1.5), float(noise_freq + max_diff * 1.5)))
        ax3.set_xlabel('Time [minutes]')
        ax3.set_title('Individual noise frequencies [Hz]')
        ax3.tick_params(labelsize=12)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        if search_individual_noise:
            found_noise_plot = np.array(found_noise, dtype=float)
            found_noise_plot[found_noise_plot == 1] = np.nan
            found_noise_plot[~np.isnan(found_noise_plot)] = noise_peaks[~np.isnan(found_noise_plot)]
            ax3.plot(chunk_indices_plot_individual, found_noise_plot, 'o', color=green)

            if nonoisehandle is not None:
                ax3.legend([nonoisehandle], ['No clear noise peak found'], edgecolor=[0.8, 0.8, 0.8])

        # Plot scores on ax4
        scores = zapline_config.get('scores', None)
        if scores is None:
            print("Error: 'scores' not provided in 'zapline_config'.")
            return

        ax4.plot(np.nanmean(scores, axis=0), color=grey)

        mean_Nremove = np.mean(NremoveFinal) + 1
        ax4.plot([mean_Nremove, mean_Nremove], ax4.get_ylim(), color=red)
        ax4.set_xlim((0.7, float(round(scores.shape[1] / 3))))
        adaptive_nremove = zapline_config.get('adaptiveNremove', False)
        noise_comp_detect_sigma = zapline_config.get('noiseCompDetectSigma', None)
        if adaptive_nremove and noise_comp_detect_sigma is not None:
            ax4.set_title(f'Mean artifact scores [a.u.], σ for detection = {noise_comp_detect_sigma}')
        else:
            ax4.set_title('Mean artifact scores [a.u.]')
        ax4.set_xlabel('Component')
        ax4.tick_params(labelsize=12)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.legend(['Mean removed'], edgecolor=[0.8, 0.8, 0.8])

        # Plot new power on ax5
        ax5.plot(f[this_freq_idx_plot], np.mean(pxx_clean_log[this_freq_idx_plot, :], axis=1), color=green)
        ax5.set_xlim((float(f[this_freq_idx_plot][0] - 0.01), float(f[this_freq_idx_plot][-1])))

        # Plot thresholds
        this_freq_idx_upper_check = zapline_config.get('thisFreqidxUppercheck', None)
        this_freq_idx_lower_check = zapline_config.get('thisFreqidxLowercheck', None)
        remaining_noise_thresh_upper = zapline_config.get('remaining_noise_thresh_upper', None)
        remaining_noise_thresh_lower = zapline_config.get('remaining_noise_thresh_lower', None)
        proportion_above_upper = zapline_config.get('proportion_above_upper', None)
        proportion_below_lower = zapline_config.get('proportion_below_lower', None)

        try:
            if this_freq_idx_upper_check is not None and this_freq_idx_lower_check is not None:
                upper_freqs = f[this_freq_idx_upper_check]
                lower_freqs = f[this_freq_idx_lower_check]
                l1, = ax5.plot(
                    [upper_freqs[0], upper_freqs[-1]],
                    [remaining_noise_thresh_upper, remaining_noise_thresh_upper],
                    color=grey
                )
                l2, = ax5.plot(
                    [lower_freqs[0], lower_freqs[-1]],
                    [remaining_noise_thresh_lower, remaining_noise_thresh_lower],
                    color=red
                )
                legend_labels = [
                    f"{round(proportion_above_upper * 100, 2)}% above",
                    f"{round(proportion_below_lower * 100, 2)}% below"
                ]
                ax5.legend([l1, l2], legend_labels, loc='upper center', edgecolor=[0.8, 0.8, 0.8])
        except Exception as e:
            print("Could not plot thresholds:", e)

        # Y-axis limits for ax5
        if remaining_noise_thresh_lower is not None and remaining_noise_thresh_upper is not None and coarse_freq_detect_power_diff is not None:
            ylim_lower = remaining_noise_thresh_lower - 0.25 * (remaining_noise_thresh_upper - remaining_noise_thresh_lower)
            ylim_upper = np.min(np.mean(pxx_raw_log[this_freq_idx_plot, :], axis=1)) + coarse_freq_detect_power_diff * 2
            ax5.set_ylim((float(ylim_lower), float(ylim_upper)))
        else:
            y_min = np.min(np.mean(pxx_clean_log[this_freq_idx_plot, :], axis=1))
            y_max = np.max(np.mean(pxx_clean_log[this_freq_idx_plot, :], axis=1))
            ax5.set_ylim((float(y_min - 0.25 * (y_max - y_min)), float(y_max + 0.25 * (y_max - y_min))))

        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power [10*log10 μV^2/Hz]')
        ax5.set_title('Cleaned spectrum')
        ax5.tick_params(labelsize=12)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

        # Plot starting spectrum on ax6
        ax6.cla()
        meanhandles, = ax6.plot(f, np.mean(pxx_raw_log, axis=1), color=grey, linewidth=1.5)
        ax6.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax6.tick_params(labelsize=12)
        ax6.set_xlabel('Frequency')
        ax6.set_ylabel('Power [10*log10 μV^2/Hz]')
        ylimits1 = ax6.get_ylim()
        ratio_noise_raw = analytics.get('ratio_noise_raw', None)
        ax6.set_title(f'Noise frequency: {noise_freq} Hz\nRatio of noise to surroundings: {ratio_noise_raw:.2f}')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)

        # Plot removed and clean spectrum on ax7
        removedhandle, = ax7.plot(f / noise_freq, np.mean(pxx_removed_log, axis=1), color=red, linewidth=1.5)
        cleanhandle, = ax7.plot(f / noise_freq, np.mean(pxx_clean_log, axis=1), color=green, linewidth=1.5)
        ax7.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax7.tick_params(labelsize=12)
        ax7.set_xlabel('Frequency (relative to noise)')
        ax7.set_ylabel('')
        ylimits2 = ax7.get_ylim()
        ylimits = [min(ylimits1[0], ylimits2[0]), max(ylimits1[1], ylimits2[1])]
        ax6.set_ylim(tuple(map(float, ylimits)))
        ax7.set_ylim(tuple(map(float, ylimits)))
        ax6.set_xlim((float(np.min(f) - np.max(f) * 0.0032), float(np.max(f))))
        ax7.set_xlim((float(np.min(f / noise_freq) - np.max(f / noise_freq) * 0.003), float(np.max(f / noise_freq))))
        proportion_removed_noise = analytics.get('proportion_removed_noise', None)
        ratio_noise_clean = analytics.get('ratio_noise_clean', None)
        ax7.set_title(f'Removed power at noise frequency: {proportion_removed_noise * 100:.2f}%\nRatio of noise to surroundings: {ratio_noise_clean:.2f}')
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)

        # Plot shaded min/max frequency areas
        minfreq = self.config.get('minfreq', None)
        maxfreq = self.config.get('maxfreq', None)
        if minfreq is not None and maxfreq is not None:
            # Shade areas in ax6
            ax6.add_patch(Rectangle((0, ylimits[0]), minfreq, ylimits[1] - ylimits[0], color='black', alpha=0.1))
            ax6.add_patch(Rectangle((maxfreq, ylimits[0]), f[-1] - maxfreq, ylimits[1] - ylimits[0], color='black', alpha=0.1))
            # Shade areas in ax7
            ax7.add_patch(Rectangle((0, ylimits[0]), minfreq / noise_freq, ylimits[1] - ylimits[0], color='black', alpha=0.1))
            ax7.add_patch(Rectangle((maxfreq / noise_freq, ylimits[0]), (f[-1] / noise_freq) - (maxfreq / noise_freq),
                                    ylimits[1] - ylimits[0], color='black', alpha=0.1))
        ax6.legend([meanhandles], ['Raw data'], edgecolor=[0.8, 0.8, 0.8])
        ax7.legend([cleanhandle, removedhandle], ['Clean data', 'Removed data'], edgecolor=[0.8, 0.8, 0.8])

        # Plot below noise on ax8
        this_freq_idx_belownoise = (f >= max(noise_freq - 11, 0)) & (f <= noise_freq - 1)
        ax8.plot(f[this_freq_idx_belownoise], np.mean(pxx_raw_log[this_freq_idx_belownoise, :], axis=1),
                color=grey, linewidth=1.5)
        ax8.plot(f[this_freq_idx_belownoise], np.mean(pxx_clean_log[this_freq_idx_belownoise, :], axis=1),
                color=green, linewidth=1.5)
        ax8.legend(['Raw data', 'Clean data'], edgecolor=[0.8, 0.8, 0.8])
        ax8.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax8.tick_params(labelsize=12)
        ax8.set_xlabel('Frequency')
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.set_xlim((float(np.min(f[this_freq_idx_belownoise])), float(np.max(f[this_freq_idx_belownoise]))))
        proportion_removed = analytics.get('proportion_removed', None)
        proportion_removed_below_noise = analytics.get('proportion_removed_below_noise', None)
        ax8.set_title(f'Removed of full spectrum: {proportion_removed * 100:.2f}%\nRemoved below noise: {proportion_removed_below_noise * 100:.2f}%')

        plt.tight_layout()
        plt.draw()
        # Save figure to a standard location (figures/zapline_results.png)
        out_dir = os.path.join(os.getcwd(), 'figures')
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        plt.savefig(os.path.join(out_dir, 'zapline_results.png'))
        return fig




    def run(self):
        self.finalize_inputs()
        zapline_config = self.config.copy()
        analytics_results = {}
        plot_handles = []
        # If no noise frequency is detected, return original data
        clean_data = self.data.copy()

        # Iterate like MATLAB: sequentially clean and optionally discover next freq after each pass
        i_noise = 0
        while i_noise < len(zapline_config['noisefreqs']):
            noise_freq = zapline_config['noisefreqs'][i_noise]
            print(f"Removing noise at {noise_freq}Hz...")
            
            if self.config['chunkLength'] != 0:
                chunk_indices = self.fixed_chunk_detection()
            else:
                chunk_indices = self.adaptive_chunk_detection(noise_freq)
            
            zapline_config['chunkIndices'] = chunk_indices
            n_chunks = len(chunk_indices) - 1
            print(f"{n_chunks} chunks will be created.")
            
            cleaning_done = False
            cleaning_too_strong_once=False
            while not cleaning_done:
                clean_data = np.zeros_like(self.data)
                scores = np.full((n_chunks, self.config['nkeep']), np.nan)
                n_remove_final = np.zeros(n_chunks)
                noise_peaks = np.zeros(n_chunks)
                found_noise = np.zeros(n_chunks)

                for i_chunk in range(n_chunks):
                    chunk = self.data[chunk_indices[i_chunk]:chunk_indices[i_chunk+1], :]
                    
                    if self.config['searchIndividualNoise']:
                        chunk_noise_freq = self.detect_chunk_noise(chunk, noise_freq)
                    else:
                        chunk_noise_freq = noise_freq

                    clean_chunk, n_remove, chunk_scores = self.apply_zapline_to_chunk(chunk, chunk_noise_freq)
                    clean_data[chunk_indices[i_chunk]:chunk_indices[i_chunk+1], :] = clean_chunk
                    
                    # Store DSS scores for this chunk (pad/truncate to configured width like MATLAB)
                    n_to_copy = min(scores.shape[1], len(chunk_scores))
                    scores[i_chunk, :n_to_copy] = chunk_scores[:n_to_copy]
                    n_remove_final[i_chunk] = n_remove
                    noise_peaks[i_chunk] = chunk_noise_freq
                    found_noise[i_chunk] = 1 if chunk_noise_freq != noise_freq else 0

                pxx_clean_log, f = self.compute_spectrum(clean_data)
                # For analytics we only need raw and clean; removed spectrum is computed for plotting later
                
                analytics = self.compute_analytics(self.pxx_raw_log, pxx_clean_log, f, noise_freq)
                cleaning_done, zapline_config, cleaning_too_strong_once = self.adaptive_cleaning(
                    clean_data, self.data, noise_freq, zapline_config, cleaning_too_strong_once
                )
                # Propagate adaptive updates into the live config used by apply_zapline_to_chunk
                self.config['noiseCompDetectSigma'] = zapline_config.get('noiseCompDetectSigma', self.config['noiseCompDetectSigma'])
                self.config['fixedNremove'] = zapline_config.get('fixedNremove', self.config['fixedNremove'])
                zapline_config['noisePeaks'] = noise_peaks
                zapline_config['scores'] = scores
            if self.config['plotResults']:
                pxx_removed_log, _ = self.compute_spectrum(self.data - clean_data)

                plot_handle = self.generate_output_figures(
                                                        data=self.data,
                                                        clean_data=clean_data,
                                                        noise_freq=noise_freq,
                                                        zapline_config=zapline_config,
                                                        pxx_raw_log=self.pxx_raw_log,
                                                        pxx_clean_log=pxx_clean_log,
                                                        pxx_removed_log=pxx_removed_log,
                                                        f=f,
                                                        analytics=analytics,
                                                        NremoveFinal=n_remove_final,
                                                        )
                if plot_handle:
                    plot_handles.append(plot_handle)


            analytics_results[f'noise_freq_{noise_freq}'] = {
                'scores': scores,
                'n_remove_final': n_remove_final,
                'noise_peaks': noise_peaks,
                'found_noise': found_noise,
                **analytics
            }

            # After finishing this frequency, optionally search for the next (MATLAB behavior)
            if zapline_config.get('automaticFreqDetection', False):
                # Start search just above the noise frequency + fine upper bound
                start_minfreq = noise_freq + zapline_config.get('detailedFreqBoundsUpper', [-0.05, 0.05])[1]
                nextfreq, _, _, _ = find_next_noisefreq(
                    pxx_clean_log,
                    f,
                    start_minfreq,
                    # Use the configured coarse threshold (MATLAB default: 4 dB)
                    zapline_config.get('coarseFreqDetectPowerDiff', 4),
                    zapline_config.get('detectionWinsize', 6),
                    zapline_config.get('maxfreq', 99),
                    zapline_config.get('coarseFreqDetectLowerPowerDiff', 1.76091259055681),
                    verbose=False
                )
                if nextfreq is not None:
                    zapline_config['noisefreqs'].append(nextfreq)

            i_noise += 1

        if self.flat_channels.size > 0:
            clean_data = self.add_back_flat_channels(clean_data)

        if self.transpose_data:
            clean_data = clean_data.T

        # If plotting requested but no noise was found/processed, create a simple 'No noise' figure
        if self.config.get('plotResults', False) and len(plot_handles) == 0:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(self.f, np.mean(self.pxx_raw_log, axis=1), color=[0.2, 0.2, 0.2])
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power [10*log10 μV^2/Hz]')
                ax.set_title('No noise found')
                # save to same path as detailed figs for consistency
                import os
                plt.tight_layout()
                out_dir = os.path.join(os.getcwd(), 'figures')
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                plt.savefig(os.path.join(out_dir, 'zapline_results.png'))
                plot_handles.append(fig)
            except Exception:
                pass

        return clean_data, zapline_config, analytics_results, plot_handles

    def add_back_flat_channels(self, clean_data):
        """
        Add back flat channels that were removed during preprocessing.

        Parameters:
            clean_data (np.ndarray): Cleaned data with flat channels removed
            
        Returns:
            np.ndarray: Cleaned data with flat channels added back as zeros
        """
        if self.flat_channels.size == 0:
            return clean_data
            
        # Create output array with original number of channels
        original_n_channels = clean_data.shape[1] + len(self.flat_channels)
        n_samples = clean_data.shape[0]
        output_data = np.zeros((n_samples, original_n_channels))
        
        # Create mask for non-flat channels
        all_channels = np.arange(original_n_channels)
        non_flat_channels = np.setdiff1d(all_channels, self.flat_channels)
        
        # Insert cleaned data back into non-flat channel positions
        output_data[:, non_flat_channels] = clean_data
        # Flat channels remain as zeros
        
        return output_data

def zapline_plus(data, sampling_rate, **kwargs):
    zp = PyZaplinePlus(data, sampling_rate, **kwargs)
    return zp.run()
