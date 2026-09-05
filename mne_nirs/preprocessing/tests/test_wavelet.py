# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_nirs.preprocessing import (
    auto_wavelet,
    motion_correct_wavelet,
    motion_detect_and_correct_wavelet,
    wavelet,
)


def _inject_spikes(raw, pick):
    """Inject two isolated single-sample spikes; return the spike amplitude."""
    sig = raw._data[pick]
    spike_amp = 20 * np.std(np.diff(sig))
    n_times = raw._data.shape[1]
    raw._data[pick, n_times // 4] += spike_amp
    raw._data[pick, n_times // 2] -= spike_amp
    return spike_amp


def test_motion_correct_wavelet_reduces_spikes_od(nirs_od):
    """Wavelet correction attenuates spike artefacts in OD data."""
    pytest.importorskip("pywt")
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    _inject_spikes(nirs_od, picks[0])

    spike_before = np.max(np.abs(np.diff(nirs_od._data[picks[0]])))
    raw_corr = motion_correct_wavelet(nirs_od, iqr=1.5)
    spike_after = np.max(np.abs(np.diff(raw_corr._data[picks[0]])))
    assert spike_after < spike_before


def test_motion_correct_wavelet_reduces_spikes_hb(nirs_hb):
    """Wavelet correction works on haemoglobin concentration data."""
    pytest.importorskip("pywt")
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_hb.info)
    _inject_spikes(nirs_hb, picks[0])

    spike_before = np.max(np.abs(np.diff(nirs_hb._data[picks[0]])))
    raw_corr = motion_correct_wavelet(nirs_hb, iqr=1.5)
    spike_after = np.max(np.abs(np.diff(raw_corr._data[picks[0]])))
    assert spike_after < spike_before


def test_motion_correct_wavelet_negative_iqr_passthrough(nirs_od):
    """Negative ``iqr`` returns the data unchanged."""
    pytest.importorskip("pywt")
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    original = nirs_od._data[picks[0]].copy()

    raw_corr = motion_correct_wavelet(nirs_od, iqr=-1)
    assert_allclose(raw_corr._data[picks[0]], original)


def test_motion_correct_wavelet_returns_copy(nirs_od):
    """Wavelet correction does not modify the input Raw in place."""
    pytest.importorskip("pywt")
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    original = nirs_od._data[picks[0]].copy()

    _ = motion_correct_wavelet(nirs_od)
    assert_allclose(nirs_od._data[picks[0]], original)


def test_motion_correct_wavelet_scale_invariant(nirs_od):
    """The correction scales with the data (the threshold is relative)."""
    pytest.importorskip("pywt")
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    _inject_spikes(nirs_od, picks[0])
    raw_scaled = nirs_od.copy()
    raw_scaled._data *= 1e-6

    corr = motion_correct_wavelet(nirs_od, iqr=1.5)._data[picks]
    corr_scaled = motion_correct_wavelet(raw_scaled, iqr=1.5)._data[picks]
    assert_allclose(corr_scaled, corr * 1e-6, rtol=1e-6, atol=1e-18)


def test_motion_correct_wavelet_transform_roundtrip(nirs_od):
    """The shift-table transform and its inverse are exact."""
    pywt = pytest.importorskip("pywt")
    from mne_nirs.preprocessing.motion_removal._wavelet import (
        _inverse_shift_table,
        _pad_to_power_2,
        _shift_table,
    )

    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    padded, n_levels = _pad_to_power_2(nirs_od._data[picks[0]])
    table = _shift_table(padded, n_levels - 4, "db2", pywt)
    assert_allclose(_inverse_shift_table(table, "db2", pywt), padded, atol=1e-12)


def test_motion_detect_and_correct_wavelet(nirs_od):
    """The combined detect+correct pipeline returns a mask and annotates."""
    pytest.importorskip("pywt")
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    # inject a spike in every channel so detection flags a shared region
    nirs_od._data[picks, n_times // 2] += 1.0

    raw_corr, mask = motion_detect_and_correct_wavelet(
        nirs_od, stdev_thresh=15.0, amp_thresh=0.1, annotate=True
    )
    assert mask.shape == (n_times,)
    assert not mask.all(), "Expected the spike region to be flagged"
    assert any(d == "BAD_motion" for d in raw_corr.annotations.description)


def test_wavelet_aliases():
    """Short aliases point to the same functions."""
    assert wavelet is motion_correct_wavelet
    assert auto_wavelet is motion_detect_and_correct_wavelet


def test_motion_correct_wavelet_wrong_type():
    """Passing a non-Raw object raises TypeError."""
    pytest.importorskip("pywt")
    with pytest.raises(TypeError):
        motion_correct_wavelet(np.zeros((10, 100)))
