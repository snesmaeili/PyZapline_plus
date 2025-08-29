# Examples

## MNE EEGBCI demo

Run the included demo to detect and remove line noise from an EEGBCI segment:

```bash
python examples/mne_eegbci_demo.py
```

This generates a diagnostic figure under `figures/zapline_results.png`.

## Synthetic data

```python
import numpy as np
from pyzaplineplus import zapline_plus

fs = 500
t = np.arange(0, 30, 1/fs)
eeg = np.random.randn(t.size, 64) * 10
eeg += 8*np.sin(2*np.pi*50*t)[:, None]

clean, cfg, analytics, plots = zapline_plus(eeg, fs, noisefreqs='line', plotResults=True)
```

