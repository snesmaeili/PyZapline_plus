# MNE Integration

```python
import mne
from pyzaplineplus import zapline_plus

raw = mne.io.read_raw_fif('sample_raw.fif', preload=True)
data = raw.get_data().T
fs = raw.info['sfreq']

clean, _, _, _ = zapline_plus(data, fs, noisefreqs='line', plotResults=True)
raw._data = clean.T
raw.save('sample_raw_cleaned.fif', overwrite=True)
```

