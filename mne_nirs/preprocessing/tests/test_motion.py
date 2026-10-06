# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import mne
import numpy as np
import pytest
from mne.datasets.testing import data_path, requires_testing_data
from numpy.testing import assert_allclose, assert_array_equal

from mne_nirs.preprocessing import (
    detect_motion_artifacts,
    motion_correct_spline,
    motion_correct_wavelet,
)

# Results of Homer3 v1.80.2 on two testing dataset files, see the gist
# https://gist.github.com/leonardozaggia/9d7b02c0fc854dd7ba0df4fbaae69a9c
REFERENCE = Path(__file__).parent / "data" / "homer3_motion_correction.npz"
testing_path = data_path(download=False)
pytestmark = pytest.mark.filterwarnings("ignore:.*only contains 2D:RuntimeWarning")


@requires_testing_data
# MNE regularizes the stored time vector of the NIRSport2 file, Homer3 does not
@pytest.mark.parametrize("key, spline_atol", [("nirsport2", 1e-5), ("nirx15", 1e-12)])
def test_motion_homer3(key, spline_atol):
    """Test detection and correction against Homer3."""
    pytest.importorskip("pywt")
    ref = np.load(REFERENCE)
    raw = mne.io.read_raw_snirf(testing_path / str(ref[f"{key}__file"]))
    raw_od = mne.preprocessing.nirs.optical_density(raw).pick("fnirs_od")
    rows = [raw_od.ch_names.index(name) for name in ref[f"{key}__ch_names"]]
    stdev_thresh, amp_thresh, smoothing, iqr = ref[f"{key}__params"]
    mask = detect_motion_artifacts(
        raw_od, stdev_thresh=stdev_thresh, amp_thresh=amp_thresh
    )
    assert not mask.all()
    assert_array_equal(mask, ref[f"{key}__mask_ch"])
    assert_array_equal(mask.all(axis=0), ref[f"{key}__mask_global"])
    corrected = motion_correct_spline(raw_od, mask=mask, smoothing=smoothing)
    assert_allclose(corrected.get_data()[rows], ref[f"{key}__spline"], atol=spline_atol)
    corrected = motion_correct_wavelet(raw_od, iqr=iqr)
    assert_allclose(corrected.get_data()[rows], ref[f"{key}__wavelet"], atol=1e-11)


@requires_testing_data
def test_motion_api():
    """Test default mask, hemoglobin data, and errors."""
    fname = testing_path / "SNIRF" / "SfNIRS" / "snirf_homer3" / "1.0.3"
    raw = mne.io.read_raw_snirf(fname / "nirx_15_3_recording.snirf").load_data()
    raw_od = mne.preprocessing.nirs.optical_density(raw)
    orig = raw_od.get_data()
    corrected = motion_correct_spline(raw_od)
    assert_array_equal(raw_od.get_data(), orig)  # a copy is returned
    mask = detect_motion_artifacts(raw_od)
    assert_array_equal(
        motion_correct_spline(raw_od, mask).get_data(), corrected.get_data()
    )
    raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    assert detect_motion_artifacts(raw_hb).shape == raw_hb.get_data().shape
    with pytest.raises(ValueError, match="between 0 and 1"):
        motion_correct_spline(raw_od, smoothing=1.5)
    with pytest.raises(ValueError, match="mask must have shape"):
        motion_correct_spline(raw_od, mask[:, :-1])
    raw_eeg = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(1, 10.0, "eeg"))
    with pytest.raises(ValueError, match="exactly one of"):
        detect_motion_artifacts(raw_eeg)
