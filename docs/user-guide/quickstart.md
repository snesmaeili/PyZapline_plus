# Quick Start

```python
import numpy as np
from pyzaplineplus import zapline_plus

# Simulated EEG (time x channels)
fs = 1000
t = np.arange(0, 10, 1/fs)
eeg = np.random.randn(t.size, 64) * 10
eeg += 5*np.sin(2*np.pi*50*t)[:, None]

clean, config, analytics, plots = zapline_plus(eeg, fs, plotResults=True)
```

See also `examples/mne_eegbci_demo.py` for an MNE dataset example.

