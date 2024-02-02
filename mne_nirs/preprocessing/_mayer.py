# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd

from mne import pick_types
from mne.io import BaseRaw
from mne.utils import _validate_type, _require_version


def quantify_mayer_fooof(raw, num_oscillations=1, centre_frequency=0.01,
                         extra_df_fields={},
                         fmin=0.001, fmax=1, tmin=0, tmax=None,
                         n_fft=400, n_overlap=200,
                         peak_width_limits=(0.5, 12.0)):
    """
    Quantify Mayer wave properties using FOOOF analysis.

    The Fitting Oscillations & One Over F (FOOOF)
    :footcite:`donoghue2020parameterizing`
    is utilised to estimate Mayer wave oscillation parameters as described in
    :footcite:`luke2021characterization`.

    The FOOOF algorithm is applied to the mean PSD estimate of the data,
    the oscillation closest to the `centre_frequency` is assumed to be the
    Mayer wave oscillation. The parameters for this oscillation are returned
    as a dataframe. You can return multiple closest oscillations to the
    centre_frequency by increasing the `num_oscillations` parameter.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    num_oscillations : number
        Number of parameterised oscilations to be returned. These are selected
        in increasing distance from the `centre_frequency`.
    centre_frequency : number
        Centre frequency of the Mayer wave.
    extra_df_fields : number
        Dictionary of values to be appended to the dataframe.
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    tmin : float | None
        Min time of interest.
    tmax : float | None
        Max time of interest.
    n_fft : int
        The length of FFT used, must be ``>= n_per_seg`` (default: 256).
        The segments will be zero-padded if ``n_fft > n_per_seg``.
        If n_per_seg is None, n_fft must be <= number of time points
        in the data.
    n_overlap : int
        The number of points of overlap between segments. Will be adjusted
        to be <= n_per_seg. The default value is 0.
    peak_width_limits : tuple of (float, float), optional, default: (0.5, 12.0)
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
        As used by FOOOF.

    Returns
    -------
    df : DataFrame
        Dataframe with columns.

    References
    ----------
    .. footbibliography::
    """
    _require_version('fooof', 'run the FOOOF algorithm.')
    _validate_type(raw, BaseRaw, 'raw')

    hbo_picks = pick_types(raw.info, fnirs='hbo')
    hbr_picks = pick_types(raw.info, fnirs='hbr')

    if (not len(hbo_picks)) & (not len(hbr_picks)):
        # It may be perfectly valid to compute this on optical density
        # or raw data, I just haven't tried this. Let me know if this works
        # for you and we can ease this restriction.
        raise RuntimeError('Mayer wave estimation should be run on '
                           'haemoglobin concentration data.')

    df = pd.DataFrame()

    for picks, chroma in zip([hbo_picks, hbr_picks], ["hbo", "hbr"]):
        if len(picks):

            fm_hbo = _run_fooof(raw.copy().pick(picks),
                                fmin=fmin, fmax=fmax,
                                tmin=tmin, tmax=tmax,
                                n_overlap=n_overlap, n_fft=n_fft,
                                peak_width_limits=peak_width_limits)

            cf, pw, bw = _process_fooof_output(fm_hbo, centre_frequency)

            data = dict()
            data["Centre Frequency"] = cf
            data["Bandwidth"] = bw
            data["Power"] = pw
            data["Chromaphore"] = chroma
            data = {**data, **extra_df_fields}

            df = pd.concat([df, pd.DataFrame(data, index=[0])],
                           ignore_index=True)

    return df


def _run_fooof(raw,
               fmin=0.001, fmax=1,
               tmin=0, tmax=None,
               n_overlap=200, n_fft=400,
               peak_width_limits=(0.5, 12.0)):
    """Prepare data for FOOOF including welch and scaling, then apply."""
    from fooof import FOOOF

    psd = raw.compute_psd(
        fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
        n_overlap=n_overlap, n_fft=n_fft)
    spectra, freqs = psd.get_data(return_freqs=True)

    # FOOOF doesn't like low frequencies, so multiple by 10.
    # This is corrected for in the output below.
    freqs = freqs * 10

    # Remember these values are really 0.001 to 1.2 Hz
    fm = FOOOF(peak_width_limits=peak_width_limits)

    # And these values are really 0.0001 to 1 Hz
    freq_range = [0.001, 10]

    fm.fit(freqs, np.mean(spectra, axis=0), freq_range)

    return fm


def _process_fooof_output(fm, centre_frequency):
    """Extract and post fix FOOOF result."""
    CFs = [d[0] for d in fm.peak_params_]

    # In this line we correct for the scaling done in _run_fooof()
    mayer_idx = _find_nearest_idx(CFs, centre_frequency * 10)

    mayer = fm.peak_params_[mayer_idx]

    cf = mayer[0] / 10  # Correct scaling
    bw = mayer[2] / 10  # Correct scaling
    pw = mayer[1]

    return cf, pw, bw


def _find_nearest_idx(a, a0):
    """Element idx in nd array `a` closest to the scalar value `a0`."""
    if isinstance(a, list):
        a = np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx
