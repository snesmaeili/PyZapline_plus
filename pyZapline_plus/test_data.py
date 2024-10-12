# Python Code to Load MATLAB Data and Compare with PyZaplinePlus
import mne
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'Qt4Agg', depending on your system
import matplotlib.pyplot as plt
import os
from pyzaplineplus import zapline_plus

# Load the EEGLAB .set file
eeglab_set_file = 'pyZapline_plus/data/P01_noRAC_slowWalk_basic_prepared.set'
raw = mne.io.read_raw_eeglab(eeglab_set_file, preload=True)

# Set channel montage (uncomment if you have a valid montage file)
# montage = mne.channels.read_custom_montage('pyZapline_plus/data/channel_locations.elc')
# raw.set_montage(montage, on_missing='ignore')

# Preprocess the data
raw.filter(l_freq=1.0, h_freq=None)  # High-pass filter to remove slow drifts
raw.resample(sfreq=250)  # Downsample to reduce computational load

# Extract data and info
data = raw.get_data()
sfreq = raw.info['sfreq']

# Add artificial line noise at 60 Hz and 70 Hz to all channels
n_samples = data.shape[1]
time = np.arange(n_samples) / sfreq
noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * time)
noise_70hz = 0.5 * np.sin(2 * np.pi * 70 * time)

# Add noise to all channels
data_with_noise = data + noise_60hz + noise_70hz
raw_with_noise = mne.io.RawArray(data_with_noise, raw.info)
# noisefreqs='line'
    # plotResults=True,   # Generate plots of the results
    # chunkLength=2,     
# Apply Zapline-plus
clean_data, zapline_config, analytics_results, plot_handles = zapline_plus(
    data, sfreq)


# Create a new Raw object with the cleaned data
cleaned_raw = mne.io.RawArray(clean_data, raw.info)

# Plot PSD before cleaning
psd_raw = raw.compute_psd(fmin=0, fmax=100)
fig_before = psd_raw.plot(average=True)
fig_before.suptitle('Before Zapline-plus')
fig_before.savefig(os.path.join('pyZapline_plus', 'before_cleaning.png'))

# Plot PSD after cleaning
psd_cleaned = cleaned_raw.compute_psd(fmin=0, fmax=100)
fig_after = psd_cleaned.plot(average=True)
fig_after.suptitle('After Zapline-plus')
fig_after.savefig(os.path.join('pyZapline_plus', 'after_cleaning.png'))

# Display the generated plots from zapline_plus
for fig in plot_handles:
    if fig:  # Ensure the figure is not None
        fig.show()
        plt.show(block=True)  # Force the display of the plot

# Quantitative comparison
line_freq = 50  # Adjust if your line noise is at 60 Hz
fmin, fmax = line_freq - 1, line_freq + 1

# Compute power at line frequency before cleaning
psds_before, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, average='mean')
power_before = np.mean(psds_before)

# Compute power at line frequency after cleaning
psds_after, freqs = mne.time_frequency.psd_welch(cleaned_raw, fmin=fmin, fmax=fmax, average='mean')
power_after = np.mean(psds_after)

print(f'Power at {line_freq} Hz before cleaning: {power_before}')
print(f'Power at {line_freq} Hz after cleaning: {power_after}')
print("Zapline-plus analytics results:")
for key, value in analytics_results.items():
    print(f"{key}: {value}")