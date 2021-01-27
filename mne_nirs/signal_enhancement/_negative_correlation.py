# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np

from mne.io import BaseRaw
from mne import pick_types
from mne.utils import _validate_type


def enhance_negative_correlation(raw):
    """
    Apply algorithm from Cui et al 2009.

    Cui et al, Functional Near Infrared Spectroscopy (NIRS) signal
    improvement based on negative correlation between oxygenated and
    deoxygenated hemoglobin dynamics, NeuroImage
    https://doi.org/10.1016/j.neuroimage.2009.11.050

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    hbo_channels = pick_types(raw.info, fnirs='hbo')
    hbr_channels = pick_types(raw.info, fnirs='hbr')

    if (not len(hbo_channels)) & (not len(hbr_channels)):
        raise RuntimeError('enhance_negative_correlation should '
                           'be run on haemoglobin data.')

    if len(hbo_channels) != len(hbr_channels):
        raise RuntimeError('Same number of hbo and hbr channels required.')

    for idx in range(len(hbo_channels)):
        if raw.info['chs'][hbo_channels[idx]]['ch_name'][:-4] != \
                raw.info['chs'][hbr_channels[idx]]['ch_name'][:-4]:
            raise RuntimeError('Channels must alternate between HBO and HBR.')

    for idx in range(len(hbo_channels)):
        hbo = raw._data[hbo_channels[idx]]
        hbr = raw._data[hbr_channels[idx]]

        hbo = hbo - np.mean(hbo)
        hbr = hbr - np.mean(hbr)

        hbo_std = np.std(hbo)
        hbr_std = np.std(hbr)

        alpha = hbo_std / hbr_std

        raw._data[hbo_channels[idx]] = 0.5 * (hbo - alpha * hbr)
        raw._data[hbr_channels[idx]] = -(1 / alpha) * \
            raw._data[hbo_channels[idx]]

    return raw
