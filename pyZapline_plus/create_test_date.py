import numpy as np
import mne
import scipy.io as sio

# Data specifications
n_channels = 64
n_samples = 60000
sfreq = 250  # Sampling frequency in Hz
times = np.arange(n_samples) / sfreq  # Time vector

# Generate random EEG data (simulating EEG signals)
rng = np.random.RandomState(42)
eeg_data = rng.randn(n_channels, n_samples) * 1e-6  # EEG data in Volts (scaled to microvolts)

# Add sinusoidal noise at specified frequencies
noise_frequencies = [15, 30, 50, 80]  # in Hz
for freq in noise_frequencies:
    sinusoid = np.sin(2 * np.pi * freq * times)
    eeg_data += (0.5e-6) * sinusoid  # Add noise with amplitude of 0.5 µV

# Create channel names
channel_names = [f'EEG {i+1}' for i in range(n_channels)]

# Create MNE Info object
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

# Set unit scaling for EEG channels to microvolts (µV)
for ch in info['chs']:
    if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH:
        ch['unit_mul'] = -6  # 10^-6 = microvolts

# Create RawArray
raw = mne.io.RawArray(eeg_data, info)

# Save the data in FIF format (MNE native format)
raw.save('synthetic_eeg_raw.fif', overwrite=True)

# Also save in .set format for EEGLAB
mne.export.export_raw('synthetic_eeg_raw.set', raw, fmt='eeglab', overwrite=True)
