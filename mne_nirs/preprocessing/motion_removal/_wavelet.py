# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Port of Homer3 (https://github.com/BUNPC/Homer3) hmrR_MotionCorrectWavelet.m
# and its helpers WT_inv.m, WaveletAnalysis.m, IWT_inv.m and
# NormalizationNoise.m. The structure mirrors the MATLAB code so that the
# output matches Homer3 to numerical precision.

import numpy as np
from mne.io import BaseRaw
from mne.preprocessing.nirs import _validate_nirs_info
from mne.utils import _validate_type, verbose

# Homer3 hard-codes the lowest wavelet scale used in the analysis.
_LOWEST_SCALE = 4


def _import_pywt():
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "PyWavelets is required for wavelet motion correction. "
            "Install it with: pip install PyWavelets"
        ) from exc
    return pywt


def _pad_to_power_2(signal):
    """Zero-pad a 1-D signal to the next power of 2.

    Returns
    -------
    padded : ndarray, shape (2**n_levels,)
        Zero-padded signal.
    n_levels : int
        ``ceil(log2(len(signal)))``, Homer3's ``N``.
    """
    original_length = len(signal)
    n_levels = int(np.ceil(np.log2(original_length))) if original_length > 1 else 1
    padded = np.zeros(2**n_levels)
    padded[:original_length] = signal
    return padded, n_levels


def _normalize_signal(signal, wavelet, pywt):
    """Scale the signal by its noise level (Homer3 ``NormalizationNoise``).

    The noise level is the mean absolute deviation of the first-level detail
    coefficients obtained by circular convolution with the quadrature mirror
    filter. The outlier threshold applied later is scale invariant, so this
    step does not change the corrected output; it is kept to mirror the
    reference implementation.

    Returns
    -------
    normalized : ndarray
        Scaled signal.
    coef : float
        Scale factor; divide by it to undo the normalization.
    """
    qmf = np.asarray(pywt.Wavelet(wavelet).dec_hi)
    n = len(signal)
    c = np.real(np.fft.ifft(np.fft.fft(signal, n) * np.fft.fft(qmf, n)))
    y_ds = c[::2]
    mad = np.mean(np.abs(y_ds - np.mean(y_ds)))  # MATLAB mad() default
    if mad != 0:
        coef = 1.0 / (1.4826 * mad)
        return signal * coef, coef
    return signal.copy(), 1.0


def _shift_table(x, n_detail_levels, wavelet, pywt):
    """Translation-invariant wavelet table (Homer3 ``WT_inv``).

    At every level each block is replaced by the DWT of the block and the DWT
    of the block shifted by one sample, both with periodic boundary handling.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Signal of length ``2**k``.
    n_detail_levels : int
        Number of detail levels, Homer3's ``N - L``.
    wavelet : str
        Orthogonal wavelet name.
    pywt : module
        The PyWavelets module.

    Returns
    -------
    table : ndarray, shape (n, n_detail_levels + 1)
        Column 0 holds the final approximation coefficients, column ``d + 1``
        the detail coefficients of level ``d`` (0 = finest).
    """
    n = len(x)
    table = np.zeros((n, n_detail_levels + 1))
    table[:, 0] = x
    for d in range(n_detail_levels):
        n_blocks = 2**d
        length = n // n_blocks
        half = length // 2
        for b in range(n_blocks):
            start = b * length
            s = table[start : start + length, 0].copy()
            s_shift = np.roll(s, 1)
            ca, cd = pywt.dwt(s, wavelet, mode="periodization")
            ca_shift, cd_shift = pywt.dwt(s_shift, wavelet, mode="periodization")
            table[start : start + half, 0] = ca
            table[start + half : start + length, 0] = ca_shift
            table[start : start + half, d + 1] = cd
            table[start + half : start + length, d + 1] = cd_shift
    return table


def _threshold_table(table, n_levels, iqr, signal_length):
    """Zero outlier detail coefficients in place (Homer3 ``WaveletAnalysis``).

    For every level and block the quartiles are computed on the coefficients
    that stem from real (unpadded) data, and coefficients outside
    ``[q1 - iqr * IQR, q3 + iqr * IQR]`` are set to zero.
    """
    n = table.shape[0]
    valid = signal_length
    for j in range(1, n_levels - _LOWEST_SCALE):
        valid //= 2
        n_blocks = 2**j
        length = n // n_blocks
        for b in range(n_blocks):
            sl = slice(b * length, (b + 1) * length)
            coeffs = table[sl, j]
            q1, q3 = np.quantile(coeffs[:valid], [0.25, 0.75], method="hazen")
            spread = q3 - q1
            outliers = (coeffs > q3 + iqr * spread) | (coeffs < q1 - iqr * spread)
            coeffs[outliers] = 0.0
            table[sl, j] = coeffs
    return table


def _inverse_shift_table(table, wavelet, pywt):
    """Invert :func:`_shift_table` (Homer3 ``IWT_inv``)."""
    n, n_cols = table.shape
    approx = table[:, 0].copy()
    for d in range(n_cols - 2, -1, -1):
        n_blocks = 2**d
        length = n // n_blocks
        half = length // 2
        for b in range(n_blocks):
            start = b * length
            cd = table[start : start + half, d + 1]
            cd_shift = table[start + half : start + length, d + 1]
            ca = approx[start : start + half]
            ca_shift = approx[start + half : start + length]
            s1 = pywt.idwt(ca, cd, wavelet, mode="periodization")
            s_shift = pywt.idwt(ca_shift, cd_shift, wavelet, mode="periodization")
            approx[start : start + length] = (s1 + np.roll(s_shift, -1)) / 2
    return approx


def _correct_channel(signal, iqr, wavelet, pywt):
    """Wavelet motion correction of one channel (one pass of Homer3's loop)."""
    padded, n_levels = _pad_to_power_2(signal)
    dc = padded.mean()
    padded -= dc
    normalized, coef = _normalize_signal(padded, wavelet, pywt)
    table = _shift_table(normalized, n_levels - _LOWEST_SCALE, wavelet, pywt)
    table = _threshold_table(table, n_levels, iqr, len(signal))
    corrected = _inverse_shift_table(table, wavelet, pywt)
    return corrected[: len(signal)] / coef + dc


@verbose
def motion_correct_wavelet(raw, iqr=1.5, wavelet="db2", *, verbose=None):
    """Apply wavelet-based motion correction to fNIRS data.

    Each channel is decomposed with a translation-invariant wavelet transform,
    detail coefficients that are outliers with respect to the interquartile
    range are set to zero, and the signal is reconstructed. The method
    targets spike artifacts :footcite:`MolaviDumont2012`.

    This is a port of Homer3's ``hmrR_MotionCorrectWavelet``
    :footcite:`HuppertEtAl2009` and reproduces its output to numerical
    precision.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    iqr : float
        Interquartile-range multiplier used as the outlier threshold for the
        wavelet coefficients. Larger values remove fewer coefficients. Set to
        a negative value to disable the correction. Default is ``1.5``.
    wavelet : str
        Orthogonal wavelet recognised by PyWavelets. Homer3 uses ``'db2'``.
        Default is ``'db2'``.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Data with wavelet motion correction applied (copy).

    See Also
    --------
    motion_detect_and_correct_wavelet : Detect, annotate and correct in one call.

    Notes
    -----
    Requires the ``PyWavelets`` package (``pip install PyWavelets``).

    There is a shorter alias ``mne_nirs.preprocessing.wavelet``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    pywt = _import_pywt()
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)
    if not len(picks):
        raise RuntimeError(
            "Wavelet motion correction should be run on optical density "
            "or hemoglobin data."
        )
    if iqr < 0:
        return raw
    for pick in picks:
        raw._data[pick] = _correct_channel(raw._data[pick].copy(), iqr, wavelet, pywt)
    return raw


# provide a short alias
wavelet = motion_correct_wavelet


def _build_motion_annotations(raw, mask):
    """Add ``BAD_motion`` annotations to *raw* for each run of ``False`` in *mask*."""
    from mne import Annotations

    bad = ~np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    starts, stops = edges[::2], edges[1::2]
    if not len(starts):
        return
    new_ann = Annotations(
        onset=raw.times[starts],
        duration=(stops - starts) / raw.info["sfreq"],
        description=["BAD_motion"] * len(starts),
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + new_ann)


@verbose
def motion_detect_and_correct_wavelet(
    raw,
    t_motion=0.5,
    t_mask=1.0,
    stdev_thresh=50.0,
    amp_thresh=5.0,
    iqr=1.5,
    wavelet="db2",
    annotate=True,
    *,
    verbose=None,
):
    """Detect motion artifacts, annotate them, and apply wavelet correction.

    Runs :func:`~mne_nirs.preprocessing.detect_motion_artifacts` followed by
    :func:`motion_correct_wavelet`, the same two-step processing stream as
    Homer3's ``hmrR_MotionArtifact`` and ``hmrR_MotionCorrectWavelet``
    :footcite:`HuppertEtAl2009`. The wavelet correction is applied to the
    whole recording; the detected segments are returned as a mask and,
    optionally, added to the annotations so that they can be inspected or
    excluded from later analysis.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    t_motion : float
        Sliding-window duration (s) for motion detection. Default is ``0.5``.
    t_mask : float
        Padding (s) applied around each detected artifact. Default is ``1.0``.
    stdev_thresh : float
        Threshold multiplier applied to the standard deviation of the first
        temporal derivative during detection. Default is ``50.0``.
    amp_thresh : float
        Absolute amplitude-change threshold (optical density or Hb units).
        Default is ``5.0``.
    iqr : float
        IQR multiplier for wavelet coefficient thresholding. Default is
        ``1.5``.
    wavelet : str
        Orthogonal wavelet recognised by PyWavelets. Default is ``'db2'``.
    annotate : bool
        If ``True`` (default), detected artifact segments are added to
        ``raw.annotations`` as ``BAD_motion`` events.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Copy of the input with wavelet motion correction applied.
    mask : ndarray of bool, shape (n_times,)
        Motion-artifact mask returned by the detection step (``tInc`` in
        Homer3). ``True`` = clean, ``False`` = artifact.

    See Also
    --------
    detect_motion_artifacts : Detection step (returns the mask only).
    motion_correct_wavelet : Correction step.

    Notes
    -----
    Requires the ``PyWavelets`` package (``pip install PyWavelets``).

    There is a shorter alias ``mne_nirs.preprocessing.auto_wavelet``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    _import_pywt()
    from ._motion_artifact import detect_motion_artifacts

    _validate_type(raw, BaseRaw, "raw")
    mask = detect_motion_artifacts(
        raw,
        t_motion=t_motion,
        t_mask=t_mask,
        stdev_thresh=stdev_thresh,
        amp_thresh=amp_thresh,
        verbose=verbose,
    )
    raw_out = raw.copy().load_data()
    if annotate:
        _build_motion_annotations(raw_out, mask)
    raw_out = motion_correct_wavelet(raw_out, iqr=iqr, wavelet=wavelet, verbose=verbose)
    return raw_out, mask


# short alias
auto_wavelet = motion_detect_and_correct_wavelet
