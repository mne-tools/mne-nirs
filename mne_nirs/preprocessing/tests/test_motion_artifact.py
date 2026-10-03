# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne_nirs.preprocessing import (
    detect_motion_artifacts,
    detect_motion_artifacts_by_channel,
    motion_artifact,
    motion_artifact_by_channel,
)
from mne_nirs.preprocessing.motion_removal._motion_artifact import (
    _mask_ch_from_annotations,
    _mask_from_annotations,
)


def _inject_spike(raw, picks, sample, amplitude=1.0):
    """Add a large single-sample spike to the given picks (copy)."""
    raw = raw.copy()
    raw._data[picks, sample] += amplitude
    return raw


# ---------------------------------------------------------------------------
# detect_motion_artifacts (global mask)
# ---------------------------------------------------------------------------


def test_detect_motion_artifacts_returns_mask(nirs_od):
    """Output is a boolean array of length n_times."""
    n_times = nirs_od._data.shape[1]
    mask = detect_motion_artifacts(nirs_od)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (n_times,)


def test_detect_motion_artifacts_clean_data_all_true(nirs_od):
    """Clean data with impossibly high thresholds yields an all-True mask."""
    mask = detect_motion_artifacts(nirs_od, stdev_thresh=1e9, amp_thresh=1e9)
    assert mask.all()


def test_detect_motion_artifacts_spike_is_flagged(nirs_od):
    """A large injected spike is flagged as bad (False)."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    spike = n_times // 3
    raw_spike = _inject_spike(nirs_od, picks, spike, amplitude=1.0)

    # amp_thresh below the spike amplitude so the (absolute) criterion fires.
    mask = detect_motion_artifacts(raw_spike, stdev_thresh=15.0, amp_thresh=0.1)
    assert not mask[spike], "Expected the spike sample to be flagged as bad"


def test_detect_motion_artifacts_dilation(nirs_od):
    """The bad region extends beyond the spike due to t_mask dilation."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    fs = nirs_od.info["sfreq"]
    spike = n_times // 2
    raw_spike = _inject_spike(nirs_od, picks, spike, amplitude=1.0)

    t_mask = 1.0
    mask = detect_motion_artifacts(
        raw_spike, t_mask=t_mask, stdev_thresh=15.0, amp_thresh=0.1
    )

    n_mask = int(np.round(t_mask * fs))
    # samples immediately before the spike are also flagged by dilation
    assert not mask[max(0, spike - n_mask) : spike].all()


def test_detect_motion_artifacts_wrong_type():
    """Passing a non-Raw object raises TypeError."""
    with pytest.raises(TypeError):
        detect_motion_artifacts(np.zeros((10, 100)))


def test_detect_motion_artifacts_requires_nirs():
    """Passing non-fNIRS Raw raises an error."""
    info = mne.create_info(["EEG1"], sfreq=10.0, ch_types=["eeg"])
    raw_eeg = mne.io.RawArray(np.zeros((1, 100)), info)
    with pytest.raises((RuntimeError, ValueError)):
        detect_motion_artifacts(raw_eeg)


def test_detect_motion_artifacts_on_hb(nirs_hb):
    """Detection also runs on haemoglobin data."""
    mask = detect_motion_artifacts(nirs_hb)
    assert mask.shape == (nirs_hb._data.shape[1],)


# ---------------------------------------------------------------------------
# detect_motion_artifacts_by_channel (per-channel mask)
# ---------------------------------------------------------------------------


def test_detect_motion_artifacts_by_channel_shape(nirs_od):
    """Output is a 2-D boolean array (n_picks, n_times)."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    mask_ch = detect_motion_artifacts_by_channel(nirs_od)
    assert mask_ch.dtype == bool
    assert mask_ch.shape == (len(picks), n_times)


def test_detect_motion_artifacts_by_channel_selective(nirs_od):
    """Only the spiked source-detector pair has bad samples; others stay clean."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    spike = n_times // 3
    raw_spike = nirs_od.copy()
    raw_spike._data[picks[0], spike] += 1.0

    mask_ch = detect_motion_artifacts_by_channel(
        raw_spike, stdev_thresh=15.0, amp_thresh=0.1
    )
    assert not mask_ch[0, spike], "Expected spiked channel flagged"
    # Both wavelengths of a pair share one mask, as in Homer3.
    assert_array_equal(mask_ch[0], mask_ch[1])
    assert mask_ch[2:].all(), "Expected other pairs to remain clean"


def test_detect_motion_artifacts_shift(nirs_od):
    """A change is attributed to the sample after it, as in Homer3."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    step = n_times // 2
    raw_step = nirs_od.copy()
    raw_step._data[picks, step:] += 1.0

    mask = detect_motion_artifacts(
        raw_step, t_motion=0.5, t_mask=0.0, stdev_thresh=1e9, amp_thresh=0.5
    )
    n_motion = int(np.round(0.5 * raw_step.info["sfreq"]))
    bad = np.flatnonzero(~mask)
    # samples step - n_motion .. step - 1 see the change within their window;
    # Homer3 flags the sample after each of them
    assert_array_equal(bad, np.arange(step - n_motion + 1, step + 1))


def test_detect_motion_artifacts_by_channel_vs_global(nirs_od):
    """Global mask is the logical AND of all per-channel masks."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    raw_spike = nirs_od.copy()
    raw_spike._data[picks[0], nirs_od._data.shape[1] // 3] += 1.0

    mask_global = detect_motion_artifacts(raw_spike, stdev_thresh=15.0, amp_thresh=0.1)
    mask_ch = detect_motion_artifacts_by_channel(
        raw_spike, stdev_thresh=15.0, amp_thresh=0.1
    )
    assert_array_equal(mask_global, mask_ch.all(axis=0))


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------


def test_aliases():
    """Short aliases point to the same functions."""
    assert motion_artifact is detect_motion_artifacts
    assert motion_artifact_by_channel is detect_motion_artifacts_by_channel


# ---------------------------------------------------------------------------
# annotation helpers
# ---------------------------------------------------------------------------


def test_mask_from_annotations(nirs_od):
    """BAD annotations produce False in the returned mask."""
    n_times = nirs_od._data.shape[1]
    fs = nirs_od.info["sfreq"]
    onset, duration = 5.0, 1.0
    nirs_od.set_annotations(mne.Annotations([onset], [duration], ["BAD_motion"]))

    mask = _mask_from_annotations(nirs_od, n_times)

    bad_start = int(np.round((onset - nirs_od.first_time) * fs))
    bad_stop = min(n_times, bad_start + int(np.round(duration * fs)))
    assert not mask[bad_start:bad_stop].any()
    assert mask[:bad_start].all()


def test_mask_ch_from_annotations_shape(nirs_od):
    """_mask_ch_from_annotations returns the right shape and matches global."""
    picks = mne.preprocessing.nirs._validate_nirs_info(nirs_od.info)
    n_times = nirs_od._data.shape[1]
    nirs_od.set_annotations(mne.Annotations([5.0], [1.0], ["BAD_motion"]))

    mask_ch = _mask_ch_from_annotations(nirs_od, len(picks), n_times)
    mask = _mask_from_annotations(nirs_od, n_times)

    assert mask_ch.shape == (len(picks), n_times)
    for row in mask_ch:
        assert_array_equal(row, mask)
