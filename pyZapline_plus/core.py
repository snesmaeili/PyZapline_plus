import numpy as np
from .noise_detection import detect_noise_frequencies
from .segmentation import adaptive_segmentation
from .cleaning import apply_zapline, detect_noise_components
from .visualization import generate_output_figures

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
            'maxsigma': kwargs.get('maxsigma', 4),
            'chunkLength': kwargs.get('chunkLength', 0),
            'minChunkLength': kwargs.get('minChunkLength', 30),
            'winSizeCompleteSpectrum': kwargs.get('winSizeCompleteSpectrum', 0),
            'nkeep': kwargs.get('nkeep', 0),
            'plotResults': kwargs.get('plotResults', True)
        }

    def run(self):
        # Step 1: Detect noise frequencies
        if not self.config['noisefreqs']:
            self.config['noisefreqs'] = detect_noise_frequencies(self.data, self.sampling_rate, self.config)

        cleaned_data = self.data.copy()
        for noise_freq in self.config['noisefreqs']:
            # Step 2: Adaptive segmentation
            chunks = adaptive_segmentation(cleaned_data, self.sampling_rate, noise_freq, self.config)

            # Step 3: Apply Zapline to each chunk
            for chunk in chunks:
                chunk_data = cleaned_data[chunk['start']:chunk['end']]
                chunk_freq = chunk['freq'] if self.config['searchIndividualNoise'] else noise_freq
                
                # Step 4: Detect noise components
                n_components = detect_noise_components(chunk_data, chunk_freq, self.config)
                
                # Step 5: Apply Zapline
                cleaned_chunk = apply_zapline(chunk_data, chunk_freq, n_components, self.config)
                
                cleaned_data[chunk['start']:chunk['end']] = cleaned_chunk

            # Step 6: Generate output figures
            if self.config['plotResults']:
                generate_output_figures(self.data, cleaned_data, noise_freq, self.config)

        return cleaned_data

def zapline_plus(data, sampling_rate, **kwargs):
    zp = PyZaplinePlus(data, sampling_rate, **kwargs)
    return zp.run()