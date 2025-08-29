# PyZaplinePlus

**Advanced Python library for automatic and adaptive removal of line noise from EEG data**

[![PyPI - Version](https://img.shields.io/pypi/v/pyzaplineplus.svg)](https://pypi.org/project/pyzaplineplus/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyzaplineplus.svg)](https://pypi.org/project/pyzaplineplus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](about/license.md)
[![CI](https://github.com/snesmaeili/PyZapline_plus/actions/workflows/ci.yml/badge.svg)](https://github.com/snesmaeili/PyZapline_plus/actions)

---

## What is PyZaplinePlus?

PyZaplinePlus is a Python adaptation of the **Zapline-plus** library, designed to automatically remove spectral peaks like line noise from EEG data while preserving the integrity of the non-noise spectrum and maintaining the data rank. 

Unlike traditional notch filters that can distort your data, PyZaplinePlus uses sophisticated spectral detection and **Denoising Source Separation (DSS)** to identify and remove line noise components adaptively, providing clean EEG signals without unnecessary loss of important neural information.

## ✨ Key Features

- **🎯 Automatic Line Noise Detection**: Detects 50 Hz, 60 Hz, or user-specified frequencies automatically
- **🧠 Adaptive Processing**: Dynamically adjusts cleaning strength to minimize negative impacts
- **📊 Advanced Algorithms**: Uses DSS combined with PCA for precise noise component identification
- **🔧 Flexible Segmentation**: Support for both fixed-length and adaptive chunk segmentation
- **📈 Comprehensive Visualization**: Built-in plotting for evaluating cleaning effectiveness
- **🐍 Easy Integration**: Works seamlessly with NumPy arrays and MNE-Python
- **⚡ Professional Quality**: Thoroughly tested, well-documented, and production-ready

## 🚀 Quick Start

### Installation

```bash
pip install pyzaplineplus
```

### Basic Usage

```python
import numpy as np
from pyzaplineplus import zapline_plus

# Your EEG data (time × channels)
data = np.random.randn(10000, 64)  # Example: 10s of 64-channel data at 1000 Hz
sampling_rate = 1000

# Clean the data - it's that simple!
cleaned_data = zapline_plus(data, sampling_rate)
```

### With MNE-Python

```python
import mne
from pyzaplineplus import zapline_plus

# Load your MNE data
raw = mne.io.read_raw_fif('your_data.fif', preload=True)
data = raw.get_data().T  # Transpose to time × channels
sampling_rate = raw.info['sfreq']

# Clean the data
cleaned_data = zapline_plus(data, sampling_rate)

# Update your MNE object
raw._data = cleaned_data.T
```

## 🎯 Why Choose PyZaplinePlus?

### Traditional Notch Filters vs. PyZaplinePlus

| Feature | Notch Filters | PyZaplinePlus |
|---------|---------------|---------------|
| **Spectral Distortion** | ❌ Can distort nearby frequencies | ✅ Preserves non-noise spectrum |
| **Adaptivity** | ❌ Fixed filtering | ✅ Adaptive to data characteristics |
| **Rank Preservation** | ❌ May reduce data rank | ✅ Maintains data rank |
| **Multiple Frequencies** | ❌ Requires multiple filters | ✅ Handles multiple frequencies simultaneously |
| **Automatic Detection** | ❌ Manual frequency specification | ✅ Automatic noise detection |

### Scientific Foundation

PyZaplinePlus is based on peer-reviewed research:

- **Zapline-plus**: [Klug & Kloosterman (2022)](https://doi.org/10.1002/hbm.25832) - *Human Brain Mapping*
- **Original ZapLine**: [de Cheveigné (2020)](https://doi.org/10.1016/j.neuroimage.2019.116356) - *NeuroImage*

## 🔬 Use Cases

- **EEG Signal Processing**: Remove power line noise from EEG recordings
- **BCI Research**: Clean data for brain-computer interface applications  
- **Clinical Neuroscience**: Preprocess data for clinical EEG analysis
- **Research Studies**: Ensure high-quality data for scientific publications
- **Real-time Applications**: Suitable for both offline and online processing

## 📖 Documentation

- **[Installation Guide](user-guide/installation.md)**: Detailed installation instructions
- **[Quick Start](user-guide/quickstart.md)**: Get up and running in minutes
- **[Examples](user-guide/examples.md)**: Comprehensive usage examples
- **[API Reference](api/core.md)**: Complete function documentation
- **[MNE Integration](user-guide/mne-integration.md)**: Working with MNE-Python

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](developer-guide/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](about/license.md) file for details.

## 📚 Citation

If you use PyZaplinePlus in your research, please cite:

```bibtex
@software{esmaeili2024pyzaplineplus,
  title = {PyZaplinePlus: Advanced Python library for automatic and adaptive removal of line noise from EEG data},
  author = {Esmaeili, Sina},
  year = {2024},
  url = {https://github.com/snesmaeili/PyZapline_plus}
}
```

And the original Zapline-plus paper:

```bibtex
@article{klug2022zapline,
  title={Zapline-plus: A Zapline extension for automatic and adaptive removal of frequency-specific noise artifacts in M/EEG},
  author={Klug, Marius and Kloosterman, Niels A},
  journal={Human Brain Mapping},
  year={2022},
  doi={10.1002/hbm.25832}
}
```

## 💬 Support

- **Documentation**: https://snesmaeili.github.io/PyZapline_plus/
- **Issues**: https://github.com/snesmaeili/PyZapline_plus/issues
- **Discussions**: https://github.com/snesmaeili/PyZapline_plus/discussions
- **Email**: [sina.esmaeili@umontreal.ca](mailto:sina.esmaeili@umontreal.ca)
