# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Port of Homer3 (https://github.com/BUNPC/Homer3) hmrR_MotionCorrectSpline.m.
# The segment bookkeeping is transcribed with Homer3's 1-based indices so that
# the output matches Homer3 to numerical precision.

import numpy as np
from mne.io import BaseRaw
from mne.preprocessing.nirs import _validate_nirs_info
from mne.utils import _validate_type, logger, verbose
from scipy.interpolate import make_smoothing_spline

from ._motion_artifact import (
    _mask_ch_from_annotations,
    detect_motion_artifacts_by_channel,
)

_DT_SHORT = 0.3  # seconds
_DT_LONG = 3.0  # seconds


def _csaps(t, y, p):
    """Cubic smoothing spline with MATLAB ``csaps`` semantics, evaluated at t.

    ``csaps`` minimises ``p * sum((y - f)**2) + (1 - p) * int(f'')**2``, which
    is :func:`scipy.interpolate.make_smoothing_spline` with
    ``lam = (1 - p) / p``. ``p = 1`` is the interpolating spline and ``p = 0``
    the least-squares line.

    ``make_smoothing_spline`` needs at least five points. Shorter segments are
    left uncorrected (a zero residual), which is what the interpolating limit
    gives.
    """
    if len(t) < 5 or p >= 1:
        return y.copy()
    if p <= 0:
        return np.polyval(np.polyfit(t, y, 1), t)
    return make_smoothing_spline(t, y, lam=(1.0 - p) / p)(t)


def _window(seg_length, fs):
    """Window (in samples) over which segment means are taken (Homer3 rule)."""
    if seg_length < _DT_SHORT * fs:
        wind = seg_length
    elif seg_length < _DT_LONG * fs:
        wind = int(np.floor(_DT_SHORT * fs))
    else:
        wind = int(np.floor(seg_length / 10))
    # Homer3 does not guard against an empty window (mean([]) = NaN); a
    # single sample is used instead.
    return max(int(wind), 1)


def _sl(first, last):
    """Python slice for the 1-based inclusive MATLAB range ``first:last``."""
    return slice(max(first, 1) - 1, last)


def _correct_channel(dod, t, fs, tinc, p):
    """Spline correction of one channel (one pass of Homer3's channel loop).

    Parameters
    ----------
    dod : ndarray, shape (n_times,)
        Optical density (or hemoglobin) of the channel.
    t : ndarray, shape (n_times,)
        Time in seconds.
    fs : float
        Sampling frequency in Hz.
    tinc : ndarray of bool, shape (n_times,)
        ``True`` = clean sample, ``False`` = motion artifact.
    p : float
        Smoothing parameter.

    Returns
    -------
    out : ndarray, shape (n_times,)
        Corrected signal.
    """
    n = len(dod)
    out = dod.copy()

    # Starts and ends of the motion segments, as 1-based indices into diff()
    # like MATLAB's find(diff(tInc) == -1) and find(diff(tInc) == 1).
    d = np.diff(tinc.astype(int))
    lst_ms = list(np.flatnonzero(d == -1) + 1)
    lst_mf = list(np.flatnonzero(d == 1) + 1)
    if not lst_mf:
        lst_mf = [n]
    if not lst_ms:
        lst_ms = [1]
    if lst_ms[0] > lst_mf[0]:
        lst_ms.insert(0, 1)
    if lst_ms[-1] > lst_mf[-1]:
        lst_mf.append(n)
    lst_ms = np.asarray(lst_ms)
    lst_mf = np.asarray(lst_mf)
    lst_ml = lst_mf - lst_ms
    nb_ma = len(lst_ml)

    # Detrend every motion segment with the smoothing spline.
    for ms, mf in zip(lst_ms, lst_mf):
        seg = _sl(ms, mf - 1)
        out[seg] = dod[seg] - _csaps(t[seg], dod[seg], p)

    # First motion segment: shift to the preceding clean segment if there is
    # one, otherwise to the following one.
    first, last = lst_ms[0], lst_mf[0] - 1
    seg = _sl(first, last)
    wind_curr = _window(lst_ml[0], fs)
    if lst_ms[0] > 1:
        wind_prev = _window(lst_ms[0] - 1, fs)
        mean_prev = out[_sl(first - wind_prev, first - 1)].mean()
        mean_curr = out[_sl(first, first + wind_curr - 1)].mean()
        out[seg] = out[seg] - mean_curr + mean_prev
    else:
        if nb_ma > 1:
            seg_next_len = lst_ms[1] - lst_mf[0] + 1
        else:
            seg_next_len = n - lst_mf[0] + 1
        wind_next = _window(seg_next_len, fs)
        mean_curr = out[_sl(last - wind_curr, last - 1)].mean()
        mean_next = out[_sl(last + 1, last + wind_next)].mean()
        out[seg] = out[seg] - mean_curr + mean_next

    # Intermediate clean and motion segments.
    for kk in range(nb_ma - 1):
        first, last = lst_mf[kk], lst_ms[kk + 1] - 1
        seg = _sl(first, last)
        seg_prev_len = lst_ml[kk]
        seg_curr_len = last - first + 1
        wind_prev = _window(seg_prev_len, fs)
        wind_curr = _window(seg_curr_len, fs)
        mean_prev = out[_sl(first - wind_prev, first - 1)].mean()
        mean_curr = dod[_sl(first, first + wind_curr - 1)].mean()
        out[seg] = dod[seg] - mean_curr + mean_prev

        first, last = lst_ms[kk + 1], lst_mf[kk + 1] - 1
        seg = _sl(first, last)
        seg_prev_len = seg_curr_len
        seg_curr_len = lst_ml[kk + 1]
        wind_prev = _window(seg_prev_len, fs)
        wind_curr = _window(seg_curr_len, fs)
        mean_prev = out[_sl(first - wind_prev, first - 1)].mean()
        mean_curr = out[_sl(first, first + wind_curr - 1)].mean()
        out[seg] = out[seg] - mean_curr + mean_prev

    # Last clean segment. Homer3 starts it one sample before the end of the
    # last motion segment.
    if lst_mf[-1] < n:
        first, last = lst_mf[-1] - 1, n
        seg = _sl(first, last)
        wind_prev = _window(lst_ml[-1], fs)
        wind_curr = _window(last - first + 1, fs)
        mean_prev = out[_sl(first - wind_prev, first - 1)].mean()
        mean_curr = dod[_sl(first, first + wind_curr - 1)].mean()
        out[seg] = dod[seg] - mean_curr + mean_prev

    return out


@verbose
def motion_correct_spline(raw, smoothing=0.99, mask=None, *, verbose=None):
    """Apply spline interpolation motion correction to fNIRS data.

    Each motion-artifact segment is detrended with a cubic smoothing spline,
    and consecutive segments are then baseline-shifted so that the signal is
    continuous across their boundaries :footcite:`ScholkmannEtAl2010`.

    This is a port of Homer3's ``hmrR_MotionCorrectSpline``
    :footcite:`HuppertEtAl2009` and reproduces its output to numerical
    precision.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    smoothing : float
        Smoothing parameter of the spline, between ``0`` and ``1`` (``p`` in
        Homer3 and MATLAB's ``csaps``). ``1`` interpolates the data (no
        correction) and ``0`` fits a straight line to each segment. Default
        is ``0.99``, the value recommended in the literature.
    mask : array-like of bool, shape (n_picks, n_times) | None
        Per-channel motion-artifact mask (``tIncCh`` in Homer3). ``True`` =
        clean sample, ``False`` = motion artifact. When ``None`` (default)
        the mask is derived automatically: existing ``BAD`` annotations on
        *raw* are used first (applied to all channels); if none are present
        :func:`detect_motion_artifacts_by_channel` is called with default
        parameters. To use custom detection parameters, call
        :func:`detect_motion_artifacts_by_channel` first and pass the
        result here.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Data with spline motion correction applied (copy).

    See Also
    --------
    detect_motion_artifacts_by_channel : Build the per-channel mask with
        custom parameters.

    Notes
    -----
    ``n_picks`` is the number of fNIRS channels returned by
    ``_validate_nirs_info``.

    Motion segments shorter than five samples are left uncorrected.

    There is a shorter alias ``mne_nirs.preprocessing.spline`` that
    can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    if not 0 <= smoothing <= 1:
        raise ValueError(f"smoothing must be between 0 and 1, got {smoothing}")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError(
            "Spline motion correction should be run on optical density "
            "or hemoglobin data."
        )

    n_times = raw._data.shape[1]
    t = raw.times
    fs = raw.info["sfreq"]

    # Resolve mask: explicit mask → BAD annotations → auto-detection
    if mask is None:
        has_bad = any(
            a["description"].upper().startswith("BAD") for a in raw.annotations
        )
        if has_bad:
            logger.info("motion_correct_spline: building mask from BAD annotations.")
            mask = _mask_ch_from_annotations(raw, len(picks), n_times)
        else:
            logger.info(
                "motion_correct_spline: no BAD annotations found, running "
                "detect_motion_artifacts_by_channel with default parameters."
            )
            mask = detect_motion_artifacts_by_channel(raw, verbose=verbose)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(picks), n_times):
        raise ValueError(
            f"mask must have shape ({len(picks)}, {n_times}), got {mask.shape}"
        )

    for ch_idx, pick in enumerate(picks):
        if mask[ch_idx].all():
            continue
        raw._data[pick] = _correct_channel(
            raw._data[pick].copy(), t, fs, mask[ch_idx], smoothing
        )

    return raw


# provide a short alias
spline = motion_correct_spline
