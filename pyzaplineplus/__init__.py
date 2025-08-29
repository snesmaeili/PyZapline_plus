"""
PyZaplinePlus: Advanced Python library for automatic and adaptive removal of line noise from EEG data.

PyZaplinePlus is a Python adaptation of the Zapline-plus library, designed to automatically
remove spectral peaks like line noise from EEG data while preserving the integrity of the
non-noise spectrum and maintaining the data rank.

Main Functions:
    zapline_plus: Main function for line noise removal
    PyZaplinePlus: Main class for advanced usage

Example:
    >>> import numpy as np
    >>> from pyzaplineplus import zapline_plus
    >>>
    >>> # Generate sample data
    >>> data = np.random.randn(10000, 64)  # 10000 samples, 64 channels
    >>> sampling_rate = 1000
    >>>
    >>> # Clean the data
    >>> cleaned_data, *_ = zapline_plus(data, sampling_rate)
"""

from ._version import __version__
from .core import PyZaplinePlus, zapline_plus
from .noise_detection import find_next_noisefreq

__all__ = [
    "PyZaplinePlus",
    "zapline_plus",
    "find_next_noisefreq",
    "__version__",
]

# Package metadata
__author__ = "Sina Esmaeili"
__email__ = "sina.esmaeili@umontreal.ca"
__license__ = "MIT"
