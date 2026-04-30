# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from mne.filter import filter_data
from mne.io import BaseRaw
from mne.preprocessing.nirs import _channel_frequencies, _validate_nirs_info
from mne.utils import _validate_type, verbose
from scipy.signal import periodogram


@verbose
def peak_power(
    raw,
    time_window=10,
    threshold=0.1,
    l_freq=0.7,
    h_freq=1.5,
    l_trans_bandwidth=0.3,
    h_trans_bandwidth=0.3,
    verbose=False,
):
    """
    Compute peak spectral power metric for each channel and time window.

    As described in [1]_ and [2]_.
    This method provides a metric of data quality along the duration of
    the measurement. The user can specify the window over which the
    metric is computed.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    time_window : number
        The duration of the window over which to calculate the metric.
        Default is 10 seconds as in PHOEBE paper.
    threshold : number
        Values below this are marked as bad and annotated in the raw file.
    %(l_freq)s
    %(h_freq)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The Raw data. Optionally annotated with bad segments.
    scores : array (n_nirs, n_windows)
        Array of peak power values.
    times : list
        List of the start and end times of each window used to compute the
        peak spectral power.

    Notes
    -----
    This implementation of peak power differs from Pollonini's original [1]_ [2]_,
    in that the original calculates peak power on raw data, whereas multiple
    types are allowed here; and while both implementations calculate a kind of
    cosine similarity, the mathematical details are different. Users are
    advised to check the results and adjust parameters as needed.

    References
    ----------
    .. [1] Pollonini L et al., “PHOEBE: a method for real time mapping of
           optodes-scalp coupling in functional near-infrared spectroscopy” in
           Biomed. Opt. Express 7, 5104-5119 (2016).
    .. [2] Hernandez, Samuel Montero, and Luca Pollonini. "NIRSplot: a tool for
           quality assessment of fNIRS scans." Optics and the Brain.
           Optical Society of America, 2020.
    """
    # Copy raw to avoid modifying original and load data into memory
    raw = raw.copy().load_data()

    # Validate that the input contains raw fNIRS data
    # Note that peak_power currently does not require a specific data type (e.g. OD)
    _validate_type(raw, BaseRaw, "raw")

    # `picks` returns a list of channels ordered alphanumerically, which may differ
    # from the order of channels in `raw`. By virtue of being sorted, channels follow
    # an ordered sequence of S-D pairs and wavelengths, e.g., S1_D1 760, S1_D1 850,
    # S1_D2 760, S1_D2 850, S2_D1 760, S2_D1 850, etc. The algorithm below relies on
    # this ordering.
    picks = _validate_nirs_info(raw.info)

    # Number of wavelengths extracted from channel names
    n_wavelengths = len(np.unique(_channel_frequencies(raw.info)))

    # Bandpass filter data to extract heartbeat-related frequencies
    # Note: filtering is applied only to the selected channels (picks),
    # with channel order preserved, regardless of how the picks are ordered.
    filtered_data = filter_data(
        raw._data,
        raw.info["sfreq"],
        l_freq,
        h_freq,
        picks=picks,
        verbose=verbose,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
    )

    samples_per_window = int(np.ceil(time_window * raw.info["sfreq"]))
    n_windows = int(np.floor(len(raw) / samples_per_window))

    scores = np.zeros((len(picks), n_windows))
    times = []

    for window in range(n_windows):
        start_sample = int(window * samples_per_window)
        end_sample = min(start_sample + samples_per_window, len(raw) - 1)

        t_start = raw.times[start_sample]
        t_stop = raw.times[end_sample]
        times.append((t_start, t_stop))

        # pair indices for all channels pairs
        pair_indices = np.triu_indices(n_wavelengths, k=1)

        for gg in range(0, len(picks), n_wavelengths):
            ch_group = picks[gg : gg + n_wavelengths]
            group_data = filtered_data[ch_group, start_sample:end_sample]

            # Calculate pairwise peak power within group
            group_data = np.array([ch / (np.std(ch) or 1) for ch in group_data])
            peak_powers = []
            for ii, jj in zip(*pair_indices):
                c = np.correlate(group_data[ii], group_data[jj], "full")
                c = c / samples_per_window
                [_, pxx] = periodogram(c, fs=raw.info["sfreq"], window="hamming")
                peak_powers.append(max(pxx))

            # Use the minimum value in the group as peak power
            pp = min(peak_powers) if peak_powers else 0.0

            # Assign the same peak power value to all channels in the group
            scores[ch_group, window] = pp

            # Add BAD_PeakPower annotation to channels if below threshold
            if (threshold is not None) & (pp < threshold):
                raw.annotations.append(
                    t_start,
                    time_window,
                    "BAD_PeakPower",
                    ch_names=[[raw.ch_names[ii] for ii in ch_group]],
                )

    return raw, scores, times
