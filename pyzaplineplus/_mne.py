from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any

from .core import zapline_plus


def apply_zapline_to_raw(
    raw,
    *,
    picks: Optional[Sequence[str]] = None,
    copy: bool = True,
    line_freqs: Optional[Sequence[float] | str] = "line",
    verbose: Optional[bool] = None,
    **zap_kwargs: Any,
):
    """
    Apply Zapline-plus to an MNE Raw.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded raw object.
    picks : array-like of str | None
        Channel names to process. Defaults to EEG channels only.
    copy : bool
        If True, operate on a copy.
    line_freqs : sequence | "line" | None
        Frequencies to target. "line" uses 50/60 depending on sampling rate and geography.
    verbose : bool | None
        If not None, temporarily override MNE's verbosity.
    **zap_kwargs
        Passed through to `zapline_plus` (e.g., adaptiveNremove, detection thresholds, plotting).

    Returns
    -------
    raw_out : mne.io.BaseRaw
        Modified Raw (copy if copy=True).
    config, analytics, figs : dict, dict, list
        Same as `zapline_plus` return (figs may be empty if plotting is disabled).
    """
    try:
        import numpy as np
        import mne  # type: ignore
        from mne.utils import use_log_level  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "apply_zapline_to_raw requires 'mne' to be installed. Install with:\n"
            "    pip install pyzaplineplus[mne]\n"
            "or\n"
            "    pip install mne\n"
            "Then re-run your code."
        ) from e

    # Validate preload
    if not getattr(raw, "preload", False):
        raise ValueError("raw must be preload=True")

    # Determine picks: default to EEG-only
    if picks is None:
        pick_idx = mne.pick_types(
            raw.info,
            eeg=True,
            meg=False,
            eog=False,
            ecg=False,
            stim=False,
            seeg=False,
            dbs=False,
            ref_meg=False,
            misc=False,
        )
        pick_names = [raw.ch_names[ii] for ii in pick_idx]
    else:
        # Validate and map names to indices
        pick_names = list(picks)
        name_to_idx = {name: ii for ii, name in enumerate(raw.ch_names)}
        missing = [nm for nm in pick_names if nm not in name_to_idx]
        if missing:
            raise ValueError(f"picks contain unknown channel names: {missing}")
        pick_idx = [name_to_idx[nm] for nm in pick_names]

    # Copy semantics
    raw_out = raw.copy() if copy else raw

    # Extract data (channels x samples), transpose to (samples x channels)
    data = raw_out.get_data(picks=pick_idx)
    orig_dtype = data.dtype
    data_T = data.T
    fs = float(raw_out.info["sfreq"])  # sampling rate

    # Prepare Zapline-plus kwargs
    zkwargs = dict(zap_kwargs)
    if line_freqs is not None:
        zkwargs["noisefreqs"] = line_freqs

    # Apply zapline_plus
    # Ensure headless plotting unless explicitly requested
    zkwargs.setdefault("plotResults", False)

    # Manage MNE verbosity context
    ctx = use_log_level("INFO") if verbose else None
    if verbose is False:
        ctx = use_log_level("WARNING")
    try:
        if ctx is not None:
            ctx.__enter__()
        clean_T, config, analytics, figs = zapline_plus(data_T, fs, **zkwargs)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    # Put cleaned data back (preserve original dtype)
    clean = clean_T.T
    if clean.dtype != orig_dtype:
        # Only cast if needed, to preserve Raw dtype
        clean = clean.astype(orig_dtype, copy=False)

    # Update only selected picks
    raw_out._data[pick_idx, :] = clean

    return raw_out, config, analytics, figs

