# Real-data Zapline-plus parity check

This directory hosts a reproducible MATLAB-versus-Python sanity check on a
Brainstorm tutorial MEG dataset that contains strong 60 Hz line noise.

## Requirements

- The ``matpy`` Conda environment (contains MNE-Python and the MATLAB Engine)
- Local MATLAB installation accessible from the MATLAB Engine

## How to run

```powershell
/mnt/c/Users/s/anaconda3/envs/matpy/python.exe comparisons/real_dataset/run_real_data_sanity.py
```

The script will

1. download ``bst_raw`` from the Brainstorm tutorials (first run only),
2. extract a 60 s segment of MEG data at 300 Hz,
3. clean it with ``PyZaplinePlus`` (Python port) and ``clean_data_with_zapline_plus`` (MATLAB), and
4. write comparison artefacts to ``comparisons/real_dataset/results``:

- ``raw_clip.npy`` – cropped raw data (samples × channels)
- ``clean_python.npy`` – cleaned output from the Python port
- ``clean_matlab.npy`` – cleaned output from the MATLAB reference
- ``psd_comparison.png`` – Welch PSD overlay (raw vs. both cleaners)
- ``metrics.json`` – summary numbers (line-bin levels, relative norms, etc.)

These artefacts can be attached to parity reports or manuscripts.
