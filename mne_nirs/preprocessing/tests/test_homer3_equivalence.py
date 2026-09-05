# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Equivalence of the motion detection and correction functions with Homer3.

Homer3 (https://github.com/BUNPC/Homer3) is the reference implementation of
these algorithms. The reference results in ``data/homer3_motion_correction.npz``
were produced by running Homer3 v1.80.2 in MATLAB R2026a on two recordings
from the MNE testing dataset. The MATLAB driver and the script that reduce
the Homer3 output to the archive are kept in a public gist linked from the
pull request that added these tests. The tests below run the mne-nirs
functions on the same files with the same parameters and require:

- identical motion masks (``hmrR_MotionArtifact`` and
  ``hmrR_MotionArtifactByChannel``),
- spline correction (``hmrR_MotionCorrectSpline``) equal to numerical
  precision when the file's time vector is regular, and to 1e-5 otherwise
  (MNE regularises the time axis, Homer3 uses the stored values),
- wavelet correction (``hmrR_MotionCorrectWavelet``) equal to numerical
  precision.
"""

import warnings
from pathlib import Path

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne_nirs.preprocessing import (
    detect_motion_artifacts,
    detect_motion_artifacts_by_channel,
    motion_correct_spline,
    motion_correct_wavelet,
)

REFERENCE = Path(__file__).parent / "data" / "homer3_motion_correction.npz"
# absolute tolerance of the spline comparison per recording
SPLINE_ATOL = {"nirsport2": 1e-5, "nirx15": 1e-12}


@pytest.fixture(scope="module")
def reference():
    if not mne.datasets.has_dataset("testing"):
        pytest.skip("Requires testing dataset")
    return np.load(REFERENCE)


def _case(reference, key):
    """Return the optical density Raw and the Homer3 results for one case."""
    root = mne.datasets.testing.data_path(download=False)
    with warnings.catch_warnings():
        # one of the files only has 2D optode positions
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_snirf(
            root / str(reference[f"{key}__file"]), verbose=False
        )
    raw_od = mne.preprocessing.nirs.optical_density(raw.load_data())
    picks = mne.preprocessing.nirs._validate_nirs_info(raw_od.info)
    raw_od.pick(picks)
    stdev_thresh, amp_thresh, smoothing, iqr = reference[f"{key}__params"]
    params = dict(stdev_thresh=stdev_thresh, amp_thresh=amp_thresh)
    ch_names = [str(n) for n in reference[f"{key}__ch_names"]]
    rows = [raw_od.ch_names.index(n) for n in ch_names]
    return raw_od, rows, params, smoothing, iqr


@pytest.mark.parametrize("key", ["nirsport2", "nirx15"])
def test_detection_matches_homer3(reference, key):
    """Global and per-channel masks are identical to Homer3."""
    raw_od, _, params, _, _ = _case(reference, key)
    mask_global = detect_motion_artifacts(raw_od, **params)
    mask_ch = detect_motion_artifacts_by_channel(raw_od, **params)
    assert not mask_global.all(), "the reference should contain artifacts"
    assert_array_equal(mask_global, reference[f"{key}__mask_global"])
    assert_array_equal(mask_ch, reference[f"{key}__mask_ch"])


@pytest.mark.parametrize("key", ["nirsport2", "nirx15"])
def test_spline_matches_homer3(reference, key):
    """Spline correction with Homer3's mask reproduces Homer3's output."""
    raw_od, rows, _, smoothing, _ = _case(reference, key)
    corrected = motion_correct_spline(
        raw_od, smoothing=smoothing, mask=reference[f"{key}__mask_ch"]
    )
    assert_allclose(
        corrected.get_data()[rows], reference[f"{key}__spline"], atol=SPLINE_ATOL[key]
    )


@pytest.mark.parametrize("key", ["nirsport2", "nirx15"])
def test_wavelet_matches_homer3(reference, key):
    """Wavelet correction reproduces Homer3's output."""
    pytest.importorskip("pywt")
    raw_od, rows, _, _, iqr = _case(reference, key)
    corrected = motion_correct_wavelet(raw_od, iqr=iqr)
    assert_allclose(
        corrected.get_data()[rows], reference[f"{key}__wavelet"], atol=1e-11
    )
