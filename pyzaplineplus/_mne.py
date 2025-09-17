"""Backward-compatible MNE integration import shims."""
from __future__ import annotations

from .integration.mne import apply_zapline_to_raw, zapline_plus_epochs, zapline_plus_raw

__all__ = [
    "apply_zapline_to_raw",
    "zapline_plus_raw",
    "zapline_plus_epochs",
]

