# PyZaplinePlus

PyZaplinePlus is a Python adaptation of the Zapline-plus library, designed to automatically remove spectral peaks like line noise from EEG data while preserving the integrity of the non-noise spectrum and maintaining the data rank. Similar to its MATLAB counterpart, PyZaplinePlus searches for noise frequencies, divides the data into spatially stable chunks, and adjusts the cleaning strength dynamically to minimize negative impacts. The package also offers detailed visualizations of the cleaning process.

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

## Installation

To install PyZaplinePlus, you can use the following command:

```sh
pip install pyzaplineplus
```

### Dependencies
PyZaplinePlus requires the following Python packages:
- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`
- `mne` (optional for EEG integration)

Make sure all dependencies are installed before running the code.

## Usage

### Quick Start
If you wish to get started right away, you can use PyZaplinePlus with any EEG data matrix and sampling rate like this:

```python
import numpy as np
from pyzaplineplus import zapline_plus

# Load your data (example)
data = np.load('synthetic_eeg_data.npy')
sampling_rate = 1000

# Clean the data
cleaned_data = zapline_plus(data, sampling_rate)
```

### Integration with MNE
If you are using the MNE-Python library, PyZaplinePlus can be easily integrated into your MNE workflow for EEG preprocessing.

```python
from mne import io
from pyzaplineplus import zapline_plus

# Load your MNE data object
raw = io.read_raw_fif('sample_raw.fif', preload=True)
data = raw.get_data()
sampling_rate = raw.info['sfreq']

# Clean the data
cleaned_data = zapline_plus(data, sampling_rate)
raw._data = cleaned_data  # Update the raw object
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

## Contributing
Contributions are welcome! Feel free to open issues for bug reports, feature requests, or general discussions. If you'd like to contribute code, fork the repository and submit a pull request.

Please ensure all contributions maintain the quality and style of the existing codebase, including writing clear commit messages and documenting new features.

## License
PyZaplinePlus is released under the MIT License. See `LICENSE` for more details.

## Acknowledgments
The PyZaplinePlus algorithm is inspired by the MATLAB-based Zapline-plus implementation, initially designed for removing line noise from EEG signals. Special thanks to the original authors and contributors who laid the foundation for this adaptive noise removal approach.

## Detailed User Guide
For a detailed guide on how to use PyZaplinePlus, including configuration options and how to interpret the cleaning plots, please refer to our GitHub wiki.

## Please Cite
If you find PyZaplinePlus useful in your research, please cite the original papers:

- Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for automatic and adaptive removal of frequency-specific noise artifacts in M/EEG. Human Brain Mapping, 1–16. https://doi.org/10.1002/hbm.25832
- de Cheveigne, A. (2020). ZapLine: a simple and effective method to remove power line artifacts. NeuroImage, 1, 1-13. https://doi.org/10.1016/j.neuroimage.2019.116356

## Contact
For further questions or support, feel free to contact us via GitHub issues or email at `sina.esmaeili@umontreal.ca`.
