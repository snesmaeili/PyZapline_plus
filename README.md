# PyZaplinePlus

**🧠 Advanced Python library for automatic and adaptive removal of line noise from EEG data**

[![PyPI - Version](https://img.shields.io/pypi/v/pyzaplineplus.svg)](https://pypi.org/project/pyzaplineplus/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyzaplineplus.svg)](https://pypi.org/project/pyzaplineplus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://snesmaeili.github.io/PyZapline_plus/)
[![CI](https://github.com/snesmaeili/PyZapline_plus/actions/workflows/ci.yml/badge.svg)](https://github.com/snesmaeili/PyZapline_plus/actions)

PyZaplinePlus is a professional Python adaptation of the **Zapline-plus** library, designed to automatically remove spectral peaks like line noise from EEG data while preserving the integrity of the non-noise spectrum and maintaining the data rank. Unlike traditional notch filters, PyZaplinePlus uses sophisticated spectral detection and **Denoising Source Separation (DSS)** to identify and remove line noise components adaptively.

## Overview

EEG signals often suffer from line noise contamination, typically from power lines (e.g., 50 Hz or 60 Hz), and other artifacts. PyZaplinePlus is an adaptive noise removal tool that uses sophisticated spectral detection and denoising source separation (DSS) to identify and remove these line components, providing clean EEG signals without unnecessary loss of important information.

The main objectives of PyZaplinePlus include:
- Accurate and adaptive removal of line noise at specified frequencies.
- Support for automatic noise frequency detection within a given range.
- Methods for handling flat channels, adaptive chunk segmentation, and detailed evaluation of the cleaning process.

## Key Features

- **Line Noise Detection:** Automatic detection of line noise frequencies (50 Hz, 60 Hz, or user-specified) to ensure optimal performance in different environments.
- **Adaptive Chunk Segmentation:** The signal can be segmented using either fixed-length or adaptive chunks, enhancing the precision of noise removal.
- **Denoising Source Separation (DSS):** Uses DSS combined with PCA to identify and remove noise components, reducing interference while maintaining data integrity.
- **Advanced Spectrum Analysis:** Refined detection mechanisms for identifying noisy components in both the coarse and fine frequency domains.
- **Visualization Support:** Built-in functionality for plotting power spectra before and after denoising to allow easy evaluation of noise removal results.

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install pyzaplineplus
```

### From Source

```bash
git clone https://github.com/snesmaeili/PyZapline_plus.git
cd PyZapline_plus
pip install -e ".[dev]"
```

### With Optional Dependencies

```bash
# For MNE-Python integration
pip install pyzaplineplus[mne]

# For development
pip install pyzaplineplus[dev]
```

### Requirements

- **Python**: 3.8 or higher
- **Core dependencies**:
  - `numpy >= 1.20.0`
  - `scipy >= 1.7.0` 
  - `scikit-learn >= 1.0.0`
  - `matplotlib >= 3.3.0`
- **Optional**:
  - `mne >= 1.0.0` (for EEG integration)

## 💡 Usage

### Quick Start

```python
import numpy as np
from pyzaplineplus import zapline_plus

# Your EEG data (time × channels)
data = np.random.randn(10000, 64)  # 10s of 64-channel data at 1000 Hz
sampling_rate = 1000

# Clean the data
cleaned_data, config, analytics, plots = zapline_plus(data, sampling_rate)

# Optional: inspect analytics keys
print('Analytics keys:', list(analytics.keys()))

# Figures (if plotResults=True) are saved under ./figures/
```

### Advanced Usage

```python
from pyzaplineplus import PyZaplinePlus

# Initialize with custom parameters
zp = PyZaplinePlus(
    data, sampling_rate,
    noisefreqs=[50, 60],      # Target 50 and 60 Hz
    minfreq=45,               # Search range: 45-65 Hz
    maxfreq=65,
    chunkLength=10,           # 10-second chunks
    adaptiveNremove=True,     # Adaptive component removal
    plotResults=True          # Generate diagnostic plots
)

# Run the cleaning process
clean_data, config, analytics, plots = zp.run()
```

### Integration with MNE-Python

```python
import mne
from pyzaplineplus import zapline_plus

# Load your MNE data
raw = mne.io.read_raw_fif('sample_raw.fif', preload=True)
data = raw.get_data().T  # Transpose to time × channels
sampling_rate = raw.info['sfreq']

# Clean the data
cleaned_data, _, _, _ = zapline_plus(data, sampling_rate)

# Update your MNE object
raw._data = cleaned_data.T
```

### Real-World Example

```python
import numpy as np
from pyzaplineplus import zapline_plus

# Simulate realistic EEG data with line noise
fs = 500  # Sampling rate
duration = 30  # seconds
n_channels = 64

# Create base EEG signal
t = np.arange(0, duration, 1/fs)
eeg = np.random.randn(len(t), n_channels) * 10  # μV

# Add 50 Hz line noise
line_noise = 5 * np.sin(2 * np.pi * 50 * t)
noisy_eeg = eeg + line_noise[:, np.newaxis]

# Remove line noise
clean_eeg, config, analytics, plots = zapline_plus(
    noisy_eeg, fs,
    noisefreqs='line',  # Automatic detection
    plotResults=True
)

print(f"Noise reduction: {analytics['noise_freq_50']['proportion_removed_noise']*100:.1f}%")
```

### Parameters
- `data`: EEG signal data, with dimensions (time, channels).
- `sampling_rate`: Sampling frequency of the EEG data.
- `noisefreqs`: Frequencies of noise to be removed, or set to `'line'` to automatically detect line noise.
- Additional parameters (keyword arguments) to fine-tune the algorithm include:
  - `minfreq`, `maxfreq`: Frequency range to search for noise.
  - `adaptiveNremove`: Whether to use adaptive DSS component removal.
  - `chunkLength`: Length of chunks for fixed segmentation (in seconds).
  - `segmentLength`: Length of segments for adaptive chunk segmentation.
  - `plotResults`: Boolean to enable or disable result visualization.

### Output
The `run()` method returns the following:
- `clean_data`: The denoised EEG data.
- `config`: The configuration used during processing.
- `analytics`: Metrics and statistics evaluating the effectiveness of noise removal.
- `plots`: Handles to the generated figures, if `plotResults=True`.

## Detailed Functionality

### Line Noise Detection
PyZaplinePlus provides both coarse and fine detection methods for line noise. The initial spectrum is analyzed to detect strong line noise candidates, and a refined frequency search is conducted to locate the noise peaks more accurately.

### Denoising Source Separation (DSS)
The core of the noise removal process utilizes DSS with Principal Component Analysis (PCA) to isolate noisy components and remove them selectively. This method ensures minimal alteration of the original signal, maintaining as much of the non-noise-related brain activity as possible.

### Adaptive Cleaning
The cleaning process includes an adaptive mechanism that adjusts based on how successful the noise removal was. If the algorithm detects insufficient cleaning, it iteratively refines the parameters to ensure effective denoising.

### Visualization and Evaluation
PyZaplinePlus can plot the power spectra of the original and cleaned data, providing an overview of the changes and confirming the effectiveness of the cleaning. It also computes analytical metrics such as noise power reduction to quantify performance.

## Use Cases
- **EEG Signal Processing:** PyZaplinePlus is especially well-suited for preprocessing EEG signals collected in environments where power line noise is present.
- **BCI Research:** Brain-computer interface studies that require real-time or offline EEG data analysis can leverage PyZaplinePlus for high-quality signal preprocessing.
- **General Biomedical Signal Denoising:** PyZaplinePlus can also be applied to other physiological signals, such as ECG or EMG, for noise suppression.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup
- Coding standards
- Testing requirements
- Pull request process

**Quick start for contributors:**

```bash
git clone https://github.com/SinaEsmaeili/PyZaplinePlus.git
cd PyZaplinePlus
pip install -e ".[dev]"
pytest  # Run tests
```

## License
PyZaplinePlus is released under the MIT License. See `LICENSE` for more details.

## Acknowledgments
The PyZaplinePlus algorithm is inspired by the MATLAB-based Zapline-plus implementation, initially designed for removing line noise from EEG signals. Special thanks to the original authors and contributors who laid the foundation for this adaptive noise removal approach.

## Documentation
For full documentation, examples, and API reference, visit:

- Full docs: https://snesmaeili.github.io/PyZapline_plus/
- Examples: https://snesmaeili.github.io/PyZapline_plus/user-guide/examples/
- API: https://snesmaeili.github.io/PyZapline_plus/api/

## Please Cite
If you find PyZaplinePlus useful in your research, please cite the original papers:

- Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for automatic and adaptive removal of frequency-specific noise artifacts in M/EEG. Human Brain Mapping, 1–16. https://doi.org/10.1002/hbm.25832
- de Cheveigne, A. (2020). ZapLine: a simple and effective method to remove power line artifacts. NeuroImage, 1, 1-13. https://doi.org/10.1016/j.neuroimage.2019.116356

## 📚 Documentation

- **[Full Documentation](https://snesmaeili.github.io/PyZapline_plus/)**
- **[Installation Guide](https://snesmaeili.github.io/PyZapline_plus/user-guide/installation/)**
- **[Examples](https://snesmaeili.github.io/PyZapline_plus/user-guide/examples/)**
- **[API Reference](https://snesmaeili.github.io/PyZapline_plus/api/)**

## 💬 Support & Community

- **📖 Documentation**: https://snesmaeili.github.io/PyZapline_plus/
- **🐛 Issues**: https://github.com/snesmaeili/PyZapline_plus/issues
- **💬 Discussions**: https://github.com/snesmaeili/PyZapline_plus/discussions
- **📧 Email**: [sina.esmaeili@umontreal.ca](mailto:sina.esmaeili@umontreal.ca)

## 🏆 Related Projects

- **[MNE-Python](https://mne.tools/)**: Comprehensive neurophysiological data analysis
- **[EEGLAB](https://eeglab.org/)**: MATLAB toolbox for EEG analysis
- **[FieldTrip](https://www.fieldtriptoolbox.org/)**: Advanced analysis of MEG, EEG, and invasive electrophysiological data
