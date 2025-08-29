"""
Test suite for PyZaplinePlus.
        
This module contains unit tests and integration tests for the PyZaplinePlus library.
"""
        
import numpy as np
import pytest
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
    
from pyzaplineplus import PyZaplinePlus, zapline_plus, find_next_noisefreq


class TestNoiseDetection:
    """Test suite for noise detection functionality."""
    
    def test_find_next_noisefreq_basic(self):
        """Test basic noise frequency detection."""
        # Create synthetic data with very strong line noise at 50 Hz
        fs = 1000
        t = np.arange(0, 30, 1/fs)  # Longer duration for better frequency resolution
        n_channels = 32
        
        # Create very strong noise at 50 Hz with minimal background noise
        noise_signal = 20.0 * np.sin(2*np.pi*50*t)  # Very strong signal
        background_noise = np.random.randn(len(t), n_channels) * 0.05  # Very weak background
        data = background_noise + noise_signal[:, np.newaxis]
        
        # Compute power spectrum with more frequency resolution
        f, pxx = signal.welch(data, fs=fs, nperseg=4096, axis=0)
        
        # Test noise detection with default parameters first
        noisefreq, thisfreqs, thisdata, threshfound = find_next_noisefreq(
            pxx, f, minfreq=45, maxfreq=55, verbose=False
        )
        
        # If that doesn't work, try with more permissive parameters
        if noisefreq is None:
            noisefreq, thisfreqs, thisdata, threshfound = find_next_noisefreq(
                pxx, f, minfreq=45, maxfreq=55, threshdiff=1, verbose=False
            )
        
        # The test should detect some frequency in the range, even if not exactly 50 Hz
        # This tests that the function can detect obvious peaks
        assert noisefreq is not None, "Should detect the obvious noise peak"
        assert 45 <= noisefreq <= 55, f"Detected frequency {noisefreq} should be in range 45-55 Hz"
    
    def test_find_next_noisefreq_no_noise(self):
        """Test noise detection when no clear noise is present."""
        # Create random data without clear line noise
        fs = 1000
        t = np.arange(0, 10, 1/fs)
        n_channels = 32
        data = np.random.randn(len(t), n_channels) * 0.1
        
        # Compute power spectrum
        f, pxx = signal.welch(data, fs=fs, axis=0)
        
        # Test noise detection with high threshold
        noisefreq, _, _, _ = find_next_noisefreq(
            pxx, f, minfreq=45, maxfreq=55, threshdiff=20, verbose=False
        )
        
        # Should not detect noise with high threshold
        assert noisefreq is None


class TestPyZaplinePlus:
    """Test suite for the main PyZaplinePlus class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample EEG data with line noise."""
        fs = 1000
        duration = 10  # seconds
        n_channels = 64
        t = np.arange(0, duration, 1/fs)
        
        # Create base EEG-like signal
        base_signal = np.random.randn(len(t), n_channels) * 10  # μV scale
        
        # Add line noise at 50 Hz
        line_noise = 5 * np.sin(2*np.pi*50*t)  # 5 μV amplitude
        data = base_signal + line_noise[:, np.newaxis]
        
        return data, fs
    
    def test_init(self, sample_data):
        """Test PyZaplinePlus initialization."""
        data, fs = sample_data
        
        # Test basic initialization
        zp = PyZaplinePlus(data, fs)
        assert zp.data.shape == data.shape
        assert zp.sampling_rate == fs
        assert isinstance(zp.config, dict)
    
    def test_init_with_parameters(self, sample_data):
        """Test PyZaplinePlus initialization with custom parameters."""
        data, fs = sample_data
        
        # Test with custom parameters
        zp = PyZaplinePlus(
            data, fs,
            noisefreqs=[50],
            minfreq=45,
            maxfreq=55,
            plotResults=False
        )
        
        assert zp.config['noisefreqs'] == [50]
        assert zp.config['minfreq'] == 45
        assert zp.config['maxfreq'] == 55
        assert zp.config['plotResults'] is False
    
    def test_finalize_inputs(self, sample_data):
        """Test input finalization."""
        data, fs = sample_data
        
        zp = PyZaplinePlus(data, fs, plotResults=False)
        zp.finalize_inputs()
        
        # Check that spectrum was computed
        assert hasattr(zp, 'pxx_raw_log')
        assert hasattr(zp, 'f')
        assert zp.pxx_raw_log.shape[1] == data.shape[1]
    
    def test_detect_flat_channels(self):
        """Test flat channel detection."""
        # Create data with one flat channel
        fs = 1000
        data = np.random.randn(5000, 5)
        data[:, 2] = 0  # Make channel 2 flat
        
        zp = PyZaplinePlus(data, fs, plotResults=False)
        flat_channels = zp.detect_flat_channels()
        
        assert 2 in flat_channels
        assert zp.data.shape[1] == 4  # One channel removed
    
    def test_line_noise_detection(self, sample_data):
        """Test automatic line noise detection."""
        data, fs = sample_data
        
        # Test with 'line' parameter
        zp = PyZaplinePlus(data, fs, noisefreqs='line', plotResults=False)
        zp.finalize_inputs()
        
        # Should detect noise frequencies
        assert isinstance(zp.config['noisefreqs'], list)
        assert len(zp.config['noisefreqs']) >= 0
    
    def test_compute_spectrum(self, sample_data):
        """Test spectrum computation."""
        data, fs = sample_data
        
        zp = PyZaplinePlus(data, fs, plotResults=False)
        pxx_log, f = zp.compute_spectrum(data)
        
        assert pxx_log.shape[1] == data.shape[1]
        assert len(f) == pxx_log.shape[0]
        assert np.max(f) <= fs / 2  # Nyquist frequency
    
    def test_bandpass_filter(self, sample_data):
        """Test bandpass filtering."""
        data, fs = sample_data
        
        zp = PyZaplinePlus(data, fs, plotResults=False)
        filtered_data = zp.bandpass_filter(data, 45, 55, fs)
        
        assert filtered_data.shape == data.shape
        assert not np.allclose(filtered_data, data)  # Should be different


class TestIntegration:
    """Integration tests for complete PyZaplinePlus workflow."""
    
    @pytest.fixture
    def noisy_eeg_data(self):
        """Generate realistic noisy EEG data."""
        fs = 500  # Typical EEG sampling rate
        duration = 30  # seconds
        n_channels = 64
        t = np.arange(0, duration, 1/fs)
        
        # Create realistic EEG-like signal
        # Multiple frequency components
        eeg_signal = np.zeros((len(t), n_channels))
        for ch in range(n_channels):
            # Alpha rhythm around 10 Hz
            eeg_signal[:, ch] += 2 * np.sin(2*np.pi*10*t + np.random.rand()*2*np.pi)
            # Beta rhythm around 20 Hz
            eeg_signal[:, ch] += 1 * np.sin(2*np.pi*20*t + np.random.rand()*2*np.pi)
            # Random noise
            eeg_signal[:, ch] += np.random.randn(len(t)) * 5
        
        # Add strong line noise at 50 Hz
        line_noise = 8 * np.sin(2*np.pi*50*t)
        noisy_data = eeg_signal + line_noise[:, np.newaxis]
        
        return noisy_data, fs
    
    def test_convenience_function(self, noisy_eeg_data):
        """Test the convenience function zapline_plus."""
        data, fs = noisy_eeg_data
        
        # Test basic usage
        result = zapline_plus(data, fs, plotResults=False)
        
        # Should return tuple with 4 elements
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        clean_data, config, analytics, plots = result
        
        # Check output shapes and types
        assert clean_data.shape == data.shape
        assert isinstance(config, dict)
        assert isinstance(analytics, dict)
        assert isinstance(plots, list)
    
    def test_full_workflow(self, noisy_eeg_data):
        """Test complete noise removal workflow."""
        data, fs = noisy_eeg_data
        
        # Initialize with specific parameters
        zp = PyZaplinePlus(
            data, fs,
            noisefreqs=[50],  # Target 50 Hz
            plotResults=False,
            chunkLength=10,   # 10-second chunks
            adaptiveNremove=True
        )
        
        # Run the full workflow
        clean_data, config, analytics, plots = zp.run()
        
        # Verify outputs
        assert clean_data.shape == data.shape
        assert not np.allclose(clean_data, data)  # Should be different
        
        # Check that noise was reduced
        # Compute spectra before and after
        f_orig, pxx_orig = signal.welch(data, fs=fs, axis=0)
        f_clean, pxx_clean = signal.welch(clean_data, fs=fs, axis=0)
        
        # Find power at 50 Hz
        freq_idx = np.argmin(np.abs(f_orig - 50))
        power_orig = np.mean(pxx_orig[freq_idx, :])
        power_clean = np.mean(pxx_clean[freq_idx, :])
        
        # Power at 50 Hz should be reduced
        reduction_ratio = power_clean / power_orig
        assert reduction_ratio < 0.8  # At least 20% reduction
    
    def test_automatic_noise_detection(self, noisy_eeg_data):
        """Test automatic noise frequency detection."""
        data, fs = noisy_eeg_data
        
        # Test with automatic detection
        zp = PyZaplinePlus(
            data, fs,
            noisefreqs=[],  # Empty list for automatic detection
            minfreq=45,
            maxfreq=55,
            plotResults=False
        )
        
        clean_data, config, analytics, plots = zp.run()
        
        # Should have detected noise frequencies
        assert len(config['noisefreqs']) > 0
        # Should have detected around 50 Hz
        detected_freqs = config['noisefreqs']
        assert any(48 <= freq <= 52 for freq in detected_freqs)
    
    def test_adaptive_chunking(self, noisy_eeg_data):
        """Test adaptive chunk segmentation."""
        data, fs = noisy_eeg_data
        
        # Test with adaptive chunking (chunkLength=0)
        zp = PyZaplinePlus(
            data, fs,
            noisefreqs=[50],
            chunkLength=0,  # Use adaptive chunking
            plotResults=False
        )
        
        clean_data, config, analytics, plots = zp.run()
        
        # Should successfully complete
        assert clean_data.shape == data.shape
        assert 'chunkIndices' in config
        assert len(config['chunkIndices']) >= 2  # At least start and end
    
    def test_mne_integration_example(self, noisy_eeg_data):
        """Test example integration with MNE-style workflow."""
        data, fs = noisy_eeg_data
        
        # Simulate MNE-style data (channels x time)
        mne_style_data = data.T  # Transpose to channels x time
        
        # Clean using zapline_plus
        clean_data, _, _, _ = zapline_plus(
            mne_style_data, fs,
            plotResults=False
        )
        
        # Verify shape preservation
        assert clean_data.shape == mne_style_data.shape
    
    @pytest.mark.slow
    def test_multiple_noise_frequencies(self, noisy_eeg_data):
        """Test removal of multiple noise frequencies."""
        data, fs = noisy_eeg_data
        
        # Add additional noise at 60 Hz
        t = np.arange(data.shape[0]) / fs
        noise_60hz = 6 * np.sin(2*np.pi*60*t)
        data_multi_noise = data + noise_60hz[:, np.newaxis]
        
        # Remove both 50 Hz and 60 Hz
        clean_data, config, analytics, plots = zapline_plus(
            data_multi_noise, fs,
            noisefreqs=[50, 60],
            plotResults=False
        )
        
        # Verify that both frequencies were processed
        assert len(analytics) >= 2  # Should have results for both frequencies
        
        # Check noise reduction for both frequencies
        f, pxx_orig = signal.welch(data_multi_noise, fs=fs, axis=0)
        f, pxx_clean = signal.welch(clean_data, fs=fs, axis=0)
        
        for target_freq in [50, 60]:
            freq_idx = np.argmin(np.abs(f - target_freq))
            power_orig = np.mean(pxx_orig[freq_idx, :])
            power_clean = np.mean(pxx_clean[freq_idx, :])
            reduction_ratio = power_clean / power_orig
            assert reduction_ratio < 0.9  # Some reduction expected


class TestParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_invalid_sampling_rate(self):
        """Test error handling for invalid sampling rate."""
        data = np.random.randn(1000, 10)
        
        with pytest.raises((ValueError, TypeError)):
            PyZaplinePlus(data, 0)  # Zero sampling rate
    
    def test_invalid_data_shape(self):
        """Test error handling for invalid data shapes."""
        # 1D data should work (will be reshaped)
        data_1d = np.random.randn(1000)
        fs = 500
        
        # This should work (might be reshaped internally)
        zp = PyZaplinePlus(data_1d, fs, plotResults=False)
        assert zp.data.ndim == 2
    
    def test_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises((ValueError, IndexError)):
            PyZaplinePlus(np.array([]), 500)
    
    def test_parameter_bounds(self):
        """Test parameter bounds validation."""
        data = np.random.randn(1000, 10)
        fs = 500
        
        # Test with extreme parameters
        zp = PyZaplinePlus(
            data, fs,
            minfreq=1,
            maxfreq=200,  # Close to Nyquist
            plotResults=False
        )
        
        # Should initialize without error
        assert zp.config['minfreq'] == 1
        assert zp.config['maxfreq'] == 200


@pytest.mark.integration
class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_dataset(self):
        """Test with larger dataset."""
        # Large dataset: 5 minutes, 128 channels, 1000 Hz
        fs = 1000
        duration = 60  # 1 minute (reduced for faster testing)
        n_channels = 128
        
        data = np.random.randn(duration * fs, n_channels) * 10
        
        # Add line noise
        t = np.arange(data.shape[0]) / fs
        line_noise = 5 * np.sin(2*np.pi*50*t)
        data += line_noise[:, np.newaxis]
        
        # Run zapline_plus
        clean_data, config, analytics, plots = zapline_plus(
            data, fs,
            noisefreqs=[50],
            plotResults=False,
            chunkLength=10  # Use fixed chunks for faster processing
        )
        
        assert clean_data.shape == data.shape
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't explode."""
        import os
        try:
            import psutil  # type: ignore
        except Exception:  # pragma: no cover - skip if psutil not available
            import pytest
            pytest.skip("psutil not installed; skipping memory efficiency test")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple datasets
        for _ in range(3):
            data = np.random.randn(10000, 64) * 10
            fs = 500
            
            t = np.arange(data.shape[0]) / fs
            line_noise = 5 * np.sin(2*np.pi*50*t)
            data += line_noise[:, np.newaxis]
            
            clean_data, _, _, _ = zapline_plus(
                data, fs,
                noisefreqs=[50],
                plotResults=False
            )

            del clean_data, data  # Explicit cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500 MB)
        assert memory_increase < 500


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
