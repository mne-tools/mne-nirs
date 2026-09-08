# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_nirs.preprocessing import motion_correct_spline, spline


def test_motion_correct_spline_reduces_step_od(nirs_od):
    """Spline correction reduces a baseline-step artefact in OD data."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]

    original = nirs_od._data[picks[0]].copy()
    shift_amp = 20 * np.max(np.abs(np.diff(nirs_od._data[picks[0]])))
    nirs_od._data[picks[0], 0:30] -= shift_amp

    mask = np.ones((len(picks), n_times), dtype=bool)
    mask[:, 0:30] = False

    raw_corr = motion_correct_spline(nirs_od, smoothing=0.99, mask=mask)

    mse_before = np.mean((nirs_od._data[picks[0]] - original) ** 2)
    mse_after = np.mean((raw_corr._data[picks[0]] - original) ** 2)
    assert mse_after < mse_before


def test_motion_correct_spline_reduces_step_hb(nirs_hb):
    """Spline correction works on haemoglobin concentration data."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_hb.info)
    n_times = nirs_hb._data.shape[1]

    original = nirs_hb._data[picks[0]].copy()
    shift_amp = 20 * np.max(np.abs(np.diff(nirs_hb._data[picks[0]])))
    nirs_hb._data[picks[0], 0:30] -= shift_amp

    mask = np.ones((len(picks), n_times), dtype=bool)
    mask[:, 0:30] = False

    raw_corr = motion_correct_spline(nirs_hb, smoothing=0.99, mask=mask)

    mse_before = np.mean((nirs_hb._data[picks[0]] - original) ** 2)
    mse_after = np.mean((raw_corr._data[picks[0]] - original) ** 2)
    assert mse_after < mse_before


def test_motion_correct_spline_constant_channels(nirs_od):
    """Spline correction does not crash on (and preserves) constant channels."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]

    nirs_od._data[picks[0]] = 0.0
    nirs_od._data[picks[1]] = 1.0

    mask = np.ones((len(picks), n_times), dtype=bool)
    mask[:, 100:120] = False

    raw_corr = motion_correct_spline(nirs_od, smoothing=0.99, mask=mask)
    assert_allclose(raw_corr._data[picks[0]], 0.0)
    assert_allclose(raw_corr._data[picks[1]], 1.0)


def test_motion_correct_spline_returns_copy(nirs_od):
    """Spline correction does not modify the input Raw in place."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    original = nirs_od._data[picks[0]].copy()

    mask = np.ones((len(picks), n_times), dtype=bool)
    mask[0, 100:130] = False

    _ = motion_correct_spline(nirs_od, smoothing=0.99, mask=mask)
    assert_allclose(nirs_od._data[picks[0]], original)


def test_motion_correct_spline_auto_mask(nirs_od):
    """With mask=None the mask is derived automatically without raising."""
    raw_corr = motion_correct_spline(nirs_od, smoothing=0.99, mask=None)
    assert raw_corr._data.shape == nirs_od._data.shape


def test_motion_correct_spline_uses_bad_annotations(nirs_od):
    """When mask is None, BAD annotations drive the correction."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    original = nirs_od._data[picks[0]].copy()
    shift_amp = 20 * np.max(np.abs(np.diff(nirs_od._data[picks[0]])))
    nirs_od._data[:, 0:30] -= shift_amp
    duration = 30 / nirs_od.info["sfreq"]
    nirs_od.set_annotations(mne.Annotations([0.0], [duration], ["BAD"]))

    raw_corr = motion_correct_spline(nirs_od, smoothing=0.99, mask=None)
    # the annotated step should be attenuated (closer to the clean signal)
    mse_before = np.mean((nirs_od._data[picks[0]] - original) ** 2)
    mse_after = np.mean((raw_corr._data[picks[0]] - original) ** 2)
    assert mse_after < mse_before


def test_motion_correct_spline_smoothing_range(nirs_od):
    """Values of smoothing outside [0, 1] raise."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        motion_correct_spline(nirs_od, smoothing=1.5)


def test_motion_correct_spline_smoothing_one_flattens(nirs_od):
    """smoothing=1 interpolates the segment, which is then flat (Homer3)."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    mask = np.ones((len(picks), n_times), dtype=bool)
    mask[:, 100:130] = False
    raw_corr = motion_correct_spline(nirs_od, smoothing=1.0, mask=mask)
    segment = raw_corr._data[picks[0], 100:128]
    assert np.ptp(segment) < 1e-12
    assert np.ptp(nirs_od._data[picks[0], 100:128]) > 1e-3


def test_motion_correct_spline_mask_shape(nirs_od):
    """A mask with the wrong shape raises."""
    with pytest.raises(ValueError, match="mask must have shape"):
        motion_correct_spline(nirs_od, mask=np.ones((2, 3), dtype=bool))


def test_spline_alias():
    """Spline is an alias for motion_correct_spline."""
    assert spline is motion_correct_spline


def test_motion_correct_spline_wrong_type():
    """Passing a non-Raw object raises TypeError."""
    with pytest.raises(TypeError):
        motion_correct_spline(np.zeros((10, 100)), smoothing=0.99)
