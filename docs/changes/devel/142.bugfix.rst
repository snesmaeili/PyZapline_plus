Fixed adaptive ``ZapLine`` refitting each segment and computing full spectra on every retry,
and ignoring settings such as ``n_harmonics``, ``nfft``, ``rank``, ``reg``, and ``whiten``
when ``min_remove`` or the cap on removed components applied.
