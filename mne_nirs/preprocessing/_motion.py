# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Adapted from Homer3 v1.80.2 (https://github.com/BUNPC/Homer3) functions
# hmrR_MotionArtifactByChannel, hmrR_MotionCorrectSpline and
# hmrR_MotionCorrectWavelet, Copyright (c) 2015, David Boas, Jay Dubb,
# Ted Huppert, released under the BSD-2-Clause license (BSD-compatible).
# The index bookkeeping mirrors the MATLAB code so that the outputs match.

import numpy as np
from mne.io import BaseRaw
from mne.preprocessing.nirs import _validate_nirs_info
from mne.utils import _validate_type, logger, verbose
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import binary_dilation


def _check_raw(raw):
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    return raw, _validate_nirs_info(raw.info)


def _detect_group(data, fs, t_motion, t_mask, stdev_thresh, amp_thresh):
    """Get the motion mask shared by a group of channels (hmrR_MotionArtifact)."""
    n = data.shape[1]
    # MATLAB round() rounds halves away from zero, NumPy to even
    n_motion = int(np.floor(t_motion * fs + 0.5))
    n_mask = int(np.floor(t_mask * fs + 0.5))
    thresh = np.diff(data, axis=1).std(axis=1, ddof=1) * stdev_thresh
    # maximum |x[t + k] - x[t]| over k = 1 .. n_motion for t = 0 .. n - 2
    max_diff = np.zeros((len(data), n - 1))
    for k in range(1, min(n_motion, n - 1) + 1):
        change = np.abs(data[:, k:] - data[:, :-k])
        max_diff[:, : n - k] = np.maximum(max_diff[:, : n - k], change)
    art = ((max_diff > thresh[:, None]) | (max_diff > amp_thresh)).any(axis=0)
    art = binary_dilation(art, structure=np.ones(2 * n_mask + 1, bool))
    # Homer3 attributes the change between t and t + k to sample t + 1
    mask = np.ones(n, bool)
    mask[1:] = ~art
    return mask


@verbose
def detect_motion_artifacts(
    raw, t_motion=0.5, t_mask=1.0, stdev_thresh=50.0, amp_thresh=5.0, *, verbose=None
):
    """Detect motion artifacts in fNIRS data.

    A sample is flagged when, within the following ``t_motion`` seconds, the
    signal changes by more than ``stdev_thresh`` times the standard deviation
    of its first derivative or by more than ``amp_thresh``. Flagged samples are
    padded by ``t_mask`` seconds on each side. The channels of a
    source-detector pair are evaluated together and share one mask. This is a
    port of Homer3's ``hmrR_MotionArtifactByChannel``
    :footcite:`HuppertEtAl2009` and gives identical masks.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    t_motion : float
        Duration (s) of the window over which changes are evaluated.
    t_mask : float
        Duration (s) to pad around each flagged sample.
    stdev_thresh : float
        Threshold as a multiple of the standard deviation of the first
        derivative.
    amp_thresh : float
        Absolute threshold on the amplitude change.
    %(verbose)s

    Returns
    -------
    mask : ndarray of bool, shape (n_channels, n_times)
        Per-channel mask (``tIncCh`` in Homer3) for the fNIRS channels, where
        ``True`` means a clean sample. Use ``mask.all(axis=0)`` to get the
        global mask of Homer3's ``hmrR_MotionArtifact`` (``tInc``).

    Notes
    -----
    The default thresholds are those of Homer3. Suitable values depend on the
    sampling rate and the units of the data.

    References
    ----------
    .. footbibliography::
    """
    raw, picks = _check_raw(raw)
    data = raw.get_data(picks)
    pairs = np.array([raw.ch_names[pick].split(" ")[0] for pick in picks])
    mask = np.ones(data.shape, bool)
    for pair in np.unique(pairs):
        idx = np.flatnonzero(pairs == pair)
        mask[idx] = _detect_group(
            data[idx], raw.info["sfreq"], t_motion, t_mask, stdev_thresh, amp_thresh
        )
    logger.info(f"Flagged {100 * (~mask).mean():0.1f}% of samples as motion")
    return mask


def _csaps(t, y, p):
    """Evaluate a cubic smoothing spline with MATLAB csaps semantics."""
    # make_smoothing_spline needs at least five points; leaving short
    # segments uncorrected is the interpolating (p = 1) limit
    if len(t) < 5 or p == 1:
        return y
    if p == 0:
        return np.polyval(np.polyfit(t, y, 1), t)
    return make_smoothing_spline(t, y, lam=(1 - p) / p)(t)


def _window(seg_length, fs):
    """Get the number of samples over which segment means are taken."""
    if seg_length < 0.3 * fs:
        wind = seg_length
    elif seg_length < 3 * fs:
        wind = int(np.floor(0.3 * fs))
    else:
        wind = int(np.floor(seg_length / 10))
    return max(int(wind), 1)  # Homer3 would take mean([]) = NaN


def _sl(first, last):
    """Convert the 1-based inclusive MATLAB range first:last to a slice."""
    return slice(max(first, 1) - 1, last)


def _shift_segment(out, src, first, last, prev_length, fs):
    """Shift src[first:last] to continue from the preceding output segment."""
    mean_prev = out[_sl(first - _window(prev_length, fs), first - 1)].mean()
    length = last - first + 1
    mean_curr = src[_sl(first, first + _window(length, fs) - 1)].mean()
    out[_sl(first, last)] = src[_sl(first, last)] - mean_curr + mean_prev
    return length


def _spline_channel(dod, t, fs, mask, p):
    """Correct one channel (one pass of hmrR_MotionCorrectSpline's loop)."""
    n = len(dod)
    # 1-based starts and ends of the motion segments
    d = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(d == -1) + 1) or [1]
    stops = list(np.flatnonzero(d == 1) + 1) or [n]
    if starts[0] > stops[0]:
        starts.insert(0, 1)
    if starts[-1] > stops[-1]:
        stops.append(n)
    out = dod.copy()
    for start, stop in zip(starts, stops):
        seg = _sl(start, stop - 1)
        out[seg] = dod[seg] - _csaps(t[seg], dod[seg], p)

    # first motion segment: match the preceding clean data, else the next
    first, last = starts[0], stops[0] - 1
    if first > 1:
        length = _shift_segment(out, out, first, last, first - 1, fs)
    else:
        next_stop = starts[1] if len(starts) > 1 else n
        mean_curr = out[_sl(last - _window(last, fs), last - 1)].mean()
        wind_next = _window(next_stop - last, fs)
        mean_next = out[_sl(last + 1, last + wind_next)].mean()
        out[_sl(first, last)] += mean_next - mean_curr
        length = last
    # following clean and motion segments, each matched to the previous one
    for kk in range(1, len(starts)):
        length = _shift_segment(out, dod, stops[kk - 1], starts[kk] - 1, length, fs)
        length = _shift_segment(out, out, starts[kk], stops[kk] - 1, length, fs)
    # Homer3 starts the last clean segment one sample early
    if stops[-1] < n:
        _shift_segment(out, dod, stops[-1] - 1, n, length, fs)
    return out


@verbose
def motion_correct_spline(raw, mask=None, smoothing=0.99, *, verbose=None):
    """Apply spline interpolation motion correction to fNIRS data.

    Each motion segment is detrended with a cubic smoothing spline and the
    segments are then shifted so that the signal is continuous across their
    boundaries :footcite:`ScholkmannEtAl2010`. This is a port of Homer3's
    ``hmrR_MotionCorrectSpline`` :footcite:`HuppertEtAl2009` and reproduces
    its output to numerical precision.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    mask : ndarray of bool, shape (n_channels, n_times) | None
        Per-channel motion mask for the fNIRS channels, where ``True`` means a
        clean sample, as returned by :func:`detect_motion_artifacts`. If
        ``None``, :func:`detect_motion_artifacts` is called with its default
        parameters.
    smoothing : float
        Smoothing parameter of the spline between 0 and 1 (``p`` in Homer3 and
        MATLAB's ``csaps``), where 1 interpolates the data (no correction) and
        0 fits a straight line.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The corrected data (a copy).

    Notes
    -----
    Motion segments shorter than five samples are not detrended.

    References
    ----------
    .. footbibliography::
    """
    raw, picks = _check_raw(raw)
    if not 0 <= smoothing <= 1:
        raise ValueError(f"smoothing must be between 0 and 1, got {smoothing}")
    if mask is None:
        mask = detect_motion_artifacts(raw)
    mask = np.asarray(mask, bool)
    if mask.shape != (len(picks), len(raw.times)):
        raise ValueError(
            f"mask must have shape {(len(picks), len(raw.times))}, got {mask.shape}"
        )
    for pick, ch_mask in zip(picks, mask):
        if not ch_mask.all():
            raw._data[pick] = _spline_channel(
                raw._data[pick], raw.times, raw.info["sfreq"], ch_mask, smoothing
            )
    return raw


# Homer3 hard-codes the coarsest scale (2 ** 4 samples) left in the analysis
_LOWEST_SCALE = 4


def _wavelet_transform(x, n_levels, pywt):
    """Translation-invariant wavelet table (Homer3's WT_inv).

    At every level each block is replaced by the DWT of the block and of the
    block circularly shifted by one sample. Column 0 holds the approximation
    and column d + 1 the details of level d (0 being the finest).
    """
    table = np.zeros(x.shape + (n_levels + 1,))
    approx = x
    for d in range(n_levels):
        blocks = approx.reshape(len(x), 2**d, -1)
        coefs = (
            pywt.dwt(b, "db2", mode="periodization")
            for b in (blocks, np.roll(blocks, 1, axis=-1))
        )
        (ca, cd), (ca_shift, cd_shift) = coefs
        approx = np.concatenate([ca, ca_shift], axis=-1).reshape(x.shape)
        table[..., d + 1] = np.concatenate([cd, cd_shift], axis=-1).reshape(x.shape)
    table[..., 0] = approx
    return table


def _inverse_wavelet_transform(table, pywt):
    """Invert _wavelet_transform (Homer3's IWT_inv)."""
    approx = table[..., 0]
    for d in range(table.shape[-1] - 2, -1, -1):
        ca = approx.reshape(len(table), 2**d, 2, -1)
        cd = table[..., d + 1].reshape(ca.shape)
        sig = pywt.idwt(ca[:, :, 0], cd[:, :, 0], "db2", mode="periodization")
        sig_shift = pywt.idwt(ca[:, :, 1], cd[:, :, 1], "db2", mode="periodization")
        approx = ((sig + np.roll(sig_shift, -1, axis=-1)) / 2).reshape(approx.shape)
    return approx


@verbose
def motion_correct_wavelet(raw, iqr=1.5, *, verbose=None):
    """Apply wavelet motion correction to fNIRS data.

    Each channel is decomposed with a translation-invariant wavelet transform,
    detail coefficients that are outliers with respect to the interquartile
    range are set to zero, and the signal is reconstructed. The method targets
    spike artifacts :footcite:`MolaviDumont2012`. This is a port of Homer3's
    ``hmrR_MotionCorrectWavelet`` :footcite:`HuppertEtAl2009` and reproduces
    its output to numerical precision.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    iqr : float
        Interquartile-range multiplier used as the outlier threshold for the
        wavelet coefficients. Larger values remove fewer coefficients.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The corrected data (a copy).

    Notes
    -----
    Requires the optional dependency ``PyWavelets``, which can be installed
    with ``pip install mne-nirs[full]``.

    References
    ----------
    .. footbibliography::
    """
    import pywt

    raw, picks = _check_raw(raw)
    data = raw.get_data(picks)
    n_times = data.shape[1]
    # zero-pad to a power of 2 and remove the mean (including the padding)
    n_levels = int(np.ceil(np.log2(n_times)))
    padded = np.zeros((len(data), 2**n_levels))
    padded[:, :n_times] = data
    dc = padded.mean(axis=1, keepdims=True)
    table = _wavelet_transform(padded - dc, n_levels - _LOWEST_SCALE, pywt)
    # zero outlier detail coefficients (Homer3's WaveletAnalysis), with
    # quartiles computed from the coefficients that stem from unpadded data
    n_valid = n_times
    for j in range(1, n_levels - _LOWEST_SCALE):
        n_valid //= 2
        coefs = table[..., j].reshape(len(data), 2**j, -1)
        q1, q3 = np.quantile(
            coefs[..., :n_valid], [0.25, 0.75], axis=-1, method="hazen", keepdims=True
        )
        spread = iqr * (q3 - q1)
        coefs[(coefs > q3 + spread) | (coefs < q1 - spread)] = 0.0
        table[..., j] = coefs.reshape(len(data), -1)
    raw._data[picks] = _inverse_wavelet_transform(table, pywt)[:, :n_times] + dc
    return raw
