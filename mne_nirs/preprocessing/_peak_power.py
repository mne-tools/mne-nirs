# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np

from scipy.signal import periodogram

from mne import pick_types
from mne.io import BaseRaw
from mne.utils import _validate_type, verbose
from mne.preprocessing.nirs import (_channel_frequencies,
                                    _check_channels_ordered)
from mne.filter import filter_data


@verbose
def peak_power(raw, time_window=10, threshold=0.1, l_freq=0.7, h_freq=1.5,
               l_trans_bandwidth=0.3, h_trans_bandwidth=0.3,
               verbose=False):
    """
    Compute peak spectral power metric from [1]_ and [2]_.

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
    scores : array (n_nirs, n_windows)
        Array of peak power values.
    times : list
        List of the start and end times of each window used to compute the
        peak spectral power.

    References
    ----------
    .. [1] Pollonini L et al., “PHOEBE: a method for real time mapping of
           optodes-scalp coupling in functional near-infrared spectroscopy” in
           Biomed. Opt. Express 7, 5104-5119 (2016).
    .. [2] Hernandez, Samuel Montero, and Luca Pollonini. "NIRSplot: a tool for
           quality assessment of fNIRS scans." Optics and the Brain.
           Optical Society of America, 2020.
    """

    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    if not len(pick_types(raw.info, fnirs='fnirs_od')):
        raise RuntimeError('Scalp coupling index '
                           'should be run on optical density data.')

    freqs = np.unique(_channel_frequencies(raw.info))
    picks = _check_channels_ordered(raw.info, freqs)

    filtered_data = filter_data(raw._data, raw.info['sfreq'], l_freq, h_freq,
                                picks=picks, verbose=verbose,
                                l_trans_bandwidth=l_trans_bandwidth,
                                h_trans_bandwidth=h_trans_bandwidth)

    window_samples = int(np.ceil(time_window * raw.info['sfreq']))
    n_windows = int(np.floor(len(raw) / window_samples))

    scores = np.zeros((len(picks), n_windows))
    times = []

    for window in range(n_windows):

        start_sample = int(window * window_samples)
        end_sample = start_sample + window_samples

        t_start = raw.times[start_sample]
        t_stop = raw.times[end_sample]
        times.append((t_start, t_stop))

        for ii in picks[::2]:

            c1 = filtered_data[ii][start_sample:end_sample]
            c2 = filtered_data[ii + 1][start_sample:end_sample]

            c1 = c1 / np.std(c1)
            c2 = c2 / np.std(c2)

            c = np.correlate(c1, c2, "full")
            c = c / (window_samples)
            [f, pxx] = periodogram(c, fs=raw.info['sfreq'], window='hamming')

            scores[ii, window] = max(pxx)
            scores[ii + 1, window] = max(pxx)

            if (threshold is not None) & (max(pxx) < threshold):
                raw.annotations.append(t_start, time_window, 'BAD_PeakPower',
                                       ch_names=[raw.ch_names[ii:ii + 2]])

    return raw, scores, times
