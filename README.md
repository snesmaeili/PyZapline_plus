Pyzapline_plus

Pyzapline_plus is a Python adaptation of the Zapline-plus library, designed to automatically remove spectral peaks like line noise from EEG data while preserving the integrity of the non-noise spectrum and maintaining the data rank. Similar to its MATLAB counterpart, Pyzapline_plus searches for noise frequencies, divides the data into spatially stable chunks, and adjusts the cleaning strength dynamically to minimize negative impacts. The package also offers detailed visualizations of the cleaning process.

Quick Start

If you wish to get started right away, you can use Pyzapline_plus with any EEG data matrix and sampling rate like this:

import numpy as np
from pyzapline_plus import zapline_plus

# Load your data (example)
data = np.load('synthetic_eeg_data.npy')
sampling_rate = 1000

# Clean the data
cleaned_data = zapline_plus(data, sampling_rate)

Integration with MNE

If you are using the MNE-Python library, Pyzapline_plus can be easily integrated into your MNE workflow for EEG preprocessing.

from mne import io
from pyzapline_plus import zapline_plus

# Load your MNE data object
raw = io.read_raw_fif('sample_raw.fif', preload=True)
data = raw.get_data()
sampling_rate = raw.info['sfreq']

# Clean the data
cleaned_data = zapline_plus(data, sampling_rate)
raw._data = cleaned_data  # Update the raw object

Detailed User Guide

For a detailed guide on how to use Pyzapline_plus, including configuration options and how to interpret the cleaning plots, please refer to our GitHub wiki.

Please Cite

If you find Pyzapline_plus useful in your research, please cite the original papers:

Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for automatic and adaptive removal of frequency-specific noise artifacts in M/EEG. Human Brain Mapping, 1–16. https://doi.org/10.1002/hbm.25832

de Cheveigne, A. (2020). ZapLine: a simple and effective method to remove power line artifacts. NeuroImage, 1, 1-13. https://doi.org/10.1016/j.neuroimage.2019.116356

Dependencies for noise tools are provided with permission by Alain de Cheveigne. For more information and additional noise removal tools, please visit the original repository: NoiseTools.

Requirements

NumPy

SciPy

Matplotlib

MNE (optional for EEG integration)
