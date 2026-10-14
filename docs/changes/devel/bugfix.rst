Fixed ``ZapLine`` leaving the line in the data when the sampling rate is not a
multiple of the line frequency (e.g. 60 Hz at 200 Hz). The fractional-period
smoother now has exact zeros at the line harmonics; the former sampled kernel
passed the line at -23 dB at 200 Hz, and that part was never cleaned. The
smoother is unchanged when the period is a whole number of samples.
