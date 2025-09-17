# MNE Integration

PyZaplinePlus provides an optional adapter to clean `mne.io.Raw` objects directly, preserving channel metadata and annotations.

Basic usage
-----------

```python
import mne
from pyzaplineplus import apply_zapline_to_raw

raw = mne.io.read_raw_fif("my_raw.fif", preload=True)
raw_clean, config, analytics, figs = apply_zapline_to_raw(
    raw,
    picks=None,           # default: EEG-only
    copy=True,            # return a modified copy
    line_freqs="line",   # auto-detect 50/60 Hz
    plotResults=False,    # optional plotting
    adaptiveNremove=True, # MATLAB parity default
)

# Save if desired
raw_clean.save("my_raw_cleaned.fif", overwrite=True)
```

Key options
-----------

- picks: list of channel names to process. Defaults to EEG-only (`mne.pick_types(eeg=True, ...)`).
- copy: True to return a modified copy; False to operate in-place.
- line_freqs: sequence of frequencies (e.g. `[50, 100]`), "line" for 50/60 autodetect, or `None` to rely on automatic detection within bounds.
- dtype: the adapter preserves the underlying `Raw` dtype (e.g. float32) when writing back.

Troubleshooting
---------------

- preload: `raw` must be `preload=True`.
- channel types: default `picks=None` processes EEG channels only. Pass explicit names to include other types.
- events/annotations: these are preserved; only the selected channels’ sample data are modified.
- no `mne` installed: install extra dependencies with `pip install pyzaplineplus[mne]`.

