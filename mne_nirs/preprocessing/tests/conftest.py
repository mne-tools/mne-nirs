# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Shared synthetic fNIRS fixtures for the motion-correction tests.

These build small, self-contained :class:`mne.io.RawArray` objects so the
tests run quickly and do not depend on downloading any dataset.
"""

import mne
import numpy as np
import pytest

SFREQ = 7.8125
N_PAIRS = 6
N_TIMES = 1200
WAVELENGTHS = (760.0, 850.0)


def _synthetic_signals(n_channels, seed=0):
    """Realistic-ish fNIRS signals: slow drift + heartbeat + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIMES) / SFREQ
    data = np.zeros((n_channels, N_TIMES))
    for i in range(n_channels):
        drift = 0.05 * np.sin(2 * np.pi * 0.01 * t + i)
        heart = 0.01 * np.sin(2 * np.pi * 1.1 * t + 0.5 * i)
        noise = 0.004 * rng.standard_normal(N_TIMES)
        data[i] = drift + heart + noise
    return data


def _make_od_raw(seed=0):
    """Synthetic optical-density Raw with source-detector geometry."""
    ch_names = []
    for s in range(1, N_PAIRS + 1):
        ch_names += [
            f"S{s}_D{s} {int(WAVELENGTHS[0])}",
            f"S{s}_D{s} {int(WAVELENGTHS[1])}",
        ]
    info = mne.create_info(ch_names, SFREQ, ["fnirs_od"] * len(ch_names))
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[9] = WAVELENGTHS[i % 2]
        s = i // 2
        loc[3:6] = [s * 0.03, 0.0, 0.0]  # source
        loc[6:9] = [s * 0.03 + 0.01, 0.0, 0.0]  # detector
        loc[0:3] = [s * 0.03 + 0.005, 0.0, 0.0]  # midpoint
        ch["loc"] = loc
    data = _synthetic_signals(len(ch_names), seed=seed)
    return mne.io.RawArray(data, info)


def _make_hb_raw(seed=1):
    """Synthetic haemoglobin (hbo/hbr) Raw."""
    ch_names = []
    ch_types = []
    for s in range(1, N_PAIRS + 1):
        ch_names += [f"S{s}_D{s} hbo", f"S{s}_D{s} hbr"]
        ch_types += ["hbo", "hbr"]
    info = mne.create_info(ch_names, SFREQ, ch_types)
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        s = i // 2
        loc[3:6] = [s * 0.03, 0.0, 0.0]
        loc[6:9] = [s * 0.03 + 0.01, 0.0, 0.0]
        loc[0:3] = [s * 0.03 + 0.005, 0.0, 0.0]
        ch["loc"] = loc
    data = _synthetic_signals(len(ch_names), seed=seed)
    return mne.io.RawArray(data, info)


@pytest.fixture(name="nirs_od")
def fixture_nirs_od():
    """Return a synthetic optical-density Raw."""
    return _make_od_raw()


@pytest.fixture(name="nirs_hb")
def fixture_nirs_hb():
    """Return a synthetic haemoglobin Raw."""
    return _make_hb_raw()
