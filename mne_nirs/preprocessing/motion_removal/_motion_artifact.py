# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Port of Homer3 (https://github.com/BUNPC/Homer3) hmrR_MotionArtifact.m and
# hmrR_MotionArtifactByChannel.m. The indexing mirrors the MATLAB code so that
# the masks match Homer3 exactly.

import numpy as np
from mne.io import BaseRaw
from mne.preprocessing.nirs import _validate_nirs_info
from mne.utils import _validate_type, logger, verbose
from scipy.ndimage import binary_dilation


def _mask_from_annotations(raw, n_times):
    """Build a global mask from ``BAD`` annotations on *raw*.

    Parameters
    ----------
    raw : instance of Raw
        Raw object whose annotations are inspected.
    n_times : int
        Number of time samples.

    Returns
    -------
    mask : ndarray of bool, shape (n_times,)
        ``True`` = clean sample, ``False`` = motion artifact.
    """
    mask = np.ones(n_times, dtype=bool)
    fs = raw.info["sfreq"]
    for ann in raw.annotations:
        if ann["description"].upper().startswith("BAD"):
            onset_sec = ann["onset"] - raw.first_time
            start = max(0, int(np.round(onset_sec * fs)))
            stop = min(n_times, start + int(np.round(ann["duration"] * fs)))
            mask[start:stop] = False
    return mask


def _mask_ch_from_annotations(raw, n_picks, n_times):
    """Build a per-channel mask tiled from ``BAD`` annotations.

    Since MNE annotations are global (not per-channel), the same mask is
    applied to every channel.

    Parameters
    ----------
    raw : instance of Raw
        Raw object whose annotations are inspected.
    n_picks : int
        Number of fNIRS channels.
    n_times : int
        Number of time samples.

    Returns
    -------
    mask : ndarray of bool, shape (n_picks, n_times)
        ``True`` = clean sample, ``False`` = motion artifact.
    """
    global_mask = _mask_from_annotations(raw, n_times)
    return np.tile(global_mask, (n_picks, 1))


def _detect_group(data, fs, t_motion, t_mask, stdev_thresh, amp_thresh):
    """Motion mask shared by a group of channels.

    This is the core of Homer3's ``hmrR_MotionArtifact``: for every sample
    the maximum absolute amplitude change over the following ``t_motion``
    seconds is computed for each channel, a sample is flagged when the change
    in any channel exceeds ``stdev_thresh`` times the standard deviation of
    that channel's first derivative or exceeds ``amp_thresh``, and flagged
    samples are padded by ``t_mask`` seconds on each side. As in Homer3 the
    change between samples ``t`` and ``t + k`` is attributed to sample
    ``t + 1``.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Signals of the channels in the group.
    fs : float
        Sampling frequency in Hz.
    t_motion : float
        Sliding window duration in seconds.
    t_mask : float
        Padding duration in seconds applied around each flagged sample.
    stdev_thresh : float
        Threshold multiplier applied to the standard deviation of the
        first derivative.
    amp_thresh : float
        Absolute amplitude-change threshold.

    Returns
    -------
    mask : ndarray of bool, shape (n_times,)
        ``True`` = clean sample, ``False`` = motion artifact.
    """
    n_ch, n = data.shape
    mask = np.ones(n, dtype=bool)
    if n < 2:
        return mask
    # MATLAB round() rounds halves away from zero, numpy to even.
    n_motion = int(np.floor(t_motion * fs + 0.5))
    n_mask = int(np.floor(t_mask * fs + 0.5))

    # Relative threshold: standard deviation (N - 1) of the first derivative.
    diff = np.diff(data, axis=1)
    std_diff = diff.std(axis=1, ddof=1) if n > 2 else np.zeros(n_ch)
    mc_thresh = std_diff * stdev_thresh

    # Maximum absolute change |x[t + k] - x[t]| over k = 1 .. n_motion, for
    # t = 0 .. n - 2 (Homer3 evaluates n - 1 rows).
    max_diff = np.zeros((n_ch, n - 1))
    for k in range(1, n_motion + 1):
        if k >= n:
            break
        change = np.abs(data[:, k:] - data[:, :-k])
        max_diff[:, : n - k] = np.maximum(max_diff[:, : n - k], change)

    art = ((max_diff > mc_thresh[:, None]) | (max_diff > amp_thresh)).any(axis=0)
    if art.any():
        art = binary_dilation(art, structure=np.ones(2 * n_mask + 1, dtype=bool))
    # Homer3: tInc(1 + bad_inds) = 0, "bad inds calculated on diff so add 1".
    mask[1:] = ~art
    return mask


def _pair_groups(raw, picks):
    """Group picks by source-detector pair, following Homer3's channel model."""
    groups = {}
    for pick in picks:
        pair = raw.ch_names[pick].split(" ")[0]
        groups.setdefault(pair, []).append(pick)
    return list(groups.values())


def _check_nirs(raw):
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)
    if not len(picks):
        raise RuntimeError(
            "Motion artifact detection should be run on optical density "
            "or hemoglobin data."
        )
    return raw, picks


@verbose
def detect_motion_artifacts(
    raw,
    t_motion=0.5,
    t_mask=1.0,
    stdev_thresh=50.0,
    amp_thresh=5.0,
    *,
    verbose=None,
):
    """Detect motion artifacts across all fNIRS channels (global mask).

    Port of Homer3's ``hmrR_MotionArtifact`` :footcite:`HuppertEtAl2009`;
    the returned mask matches Homer3's ``tInc`` exactly.

    For each channel the maximum absolute amplitude change over a window of
    ``t_motion`` seconds is computed. If, at any sample, this change exceeds
    ``stdev_thresh`` times the standard deviation of the signal's first
    derivative **or** exceeds ``amp_thresh``, that sample is flagged as a
    motion artifact. Flagged samples are dilated by ``t_mask`` seconds on
    each side. A sample is clean only if **every** channel is clean at that
    time point.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    t_motion : float
        Duration (s) of the sliding window. Default is ``0.5``.
    t_mask : float
        Duration (s) to pad around each detected artifact. Default is
        ``1.0``.
    stdev_thresh : float
        Threshold multiplier applied to the standard deviation of the first
        temporal derivative. Default is ``50.0``.
    amp_thresh : float
        Absolute amplitude-change threshold. Default is ``5.0``.
    %(verbose)s

    Returns
    -------
    mask : ndarray of bool, shape (n_times,)
        Global motion-artifact mask (``tInc`` in Homer3). ``True`` = clean
        sample, ``False`` = motion artifact.

    See Also
    --------
    detect_motion_artifacts_by_channel : Per-channel version.
    motion_correct_spline : Spline-based correction.
    motion_correct_wavelet : Wavelet-based correction.

    Notes
    -----
    The default thresholds are those of Homer3. Suitable values depend on the
    sampling rate and the units of the data.

    There is a shorter alias ``mne_nirs.preprocessing.motion_artifact``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    raw, picks = _check_nirs(raw)
    n_times = raw._data.shape[1]
    fs = raw.info["sfreq"]
    mask = _detect_group(
        raw._data[picks], fs, t_motion, t_mask, stdev_thresh, amp_thresh
    )
    n_bad = int((~mask).sum())
    logger.info(
        "Detected %d bad samples (%.1f s) out of %d (%.1f%%).",
        n_bad,
        n_bad / fs,
        n_times,
        n_bad / n_times * 100,
    )
    return mask


@verbose
def detect_motion_artifacts_by_channel(
    raw,
    t_motion=0.5,
    t_mask=1.0,
    stdev_thresh=50.0,
    amp_thresh=5.0,
    *,
    verbose=None,
):
    """Detect motion artifacts per source-detector pair (channel-wise mask).

    Port of Homer3's ``hmrR_MotionArtifactByChannel``
    :footcite:`HuppertEtAl2009`; the returned mask matches Homer3's
    ``tIncCh`` exactly.

    Same detection procedure as :func:`detect_motion_artifacts`, but applied
    separately to each source-detector pair. As in Homer3 the channels of a
    pair (both wavelengths, or HbO and HbR) are evaluated together and share
    one mask.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    t_motion : float
        Duration (s) of the sliding window. Default is ``0.5``.
    t_mask : float
        Duration (s) to pad around each detected artifact. Default is
        ``1.0``.
    stdev_thresh : float
        Threshold multiplier applied to the standard deviation of the first
        temporal derivative. Default is ``50.0``.
    amp_thresh : float
        Absolute amplitude-change threshold. Default is ``5.0``.
    %(verbose)s

    Returns
    -------
    mask : ndarray of bool, shape (n_picks, n_times)
        Per-channel motion-artifact mask (``tIncCh`` in Homer3). ``True`` =
        clean sample, ``False`` = motion artifact. ``n_picks`` is the number
        of fNIRS channels returned by ``_validate_nirs_info``.

    See Also
    --------
    detect_motion_artifacts : Global (single-mask) version.
    motion_correct_spline : Spline-based correction that accepts this mask.

    Notes
    -----
    There is a shorter alias
    ``mne_nirs.preprocessing.motion_artifact_by_channel``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    raw, picks = _check_nirs(raw)
    n_times = raw._data.shape[1]
    fs = raw.info["sfreq"]
    row = {pick: i for i, pick in enumerate(picks)}
    mask = np.ones((len(picks), n_times), dtype=bool)
    for group in _pair_groups(raw, picks):
        group_mask = _detect_group(
            raw._data[group], fs, t_motion, t_mask, stdev_thresh, amp_thresh
        )
        for pick in group:
            mask[row[pick]] = group_mask
    return mask


# short aliases
motion_artifact = detect_motion_artifacts
motion_artifact_by_channel = detect_motion_artifacts_by_channel
