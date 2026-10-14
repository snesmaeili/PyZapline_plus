Fixed ``ZapLine`` leaving the line in the data when the sampling rate is not a
multiple of the line frequency (e.g. 60 Hz at 200 Hz), and ``n_select="auto"``
removing nearly every component when one component has almost no line power,
such as a strong alpha rhythm.
