import numpy as np
from .noise_detection import find_next_noisefreq
from .segmentation import adaptive_segmentation
from .cleaning import apply_zapline, detect_noise_components
from .visualization import generate_output_figures
from scipy import signal
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
        
    def adaptive_chunk_detection(self):
        pass

    def apply_zapline_to_chunk(self, chunk, noise_freq):
        pass

    def compute_analytics(self, pxx_raw, pxx_clean, f, noise_freq):
        pass

    def adaptive_cleaning(self, clean_data, raw_data, noise_freq):
        pass

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
                plot_handle = generate_output_figures(self.data, clean_data, noise_freq, zapline_config)
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

    def fixed_chunk_detection(self):
        pass

    def detect_chunk_noise(self, chunk, noise_freq):
        pass

    def add_back_flat_channels(self, clean_data):
        pass

def zapline_plus(data, sampling_rate, **kwargs):
    zp = PyZaplinePlus(data, sampling_rate, **kwargs)
    return zp.run()