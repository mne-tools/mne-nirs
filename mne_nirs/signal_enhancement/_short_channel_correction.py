# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from mne.io import BaseRaw
from mne import pick_types
from mne.utils import _validate_type
from mne.preprocessing.nirs import source_detector_distances


def short_channel_regression(raw, max_dist=0.01):
    """
    Short channel regression based on nearest channel.

    Fabbri, Francesco, et al. "Optical measurements of absorption changes in
    two-layered diffusive media."
    Physics in Medicine & Biology 49.7 (2004): 1183.

    Saager, Rolf B., and Andrew J. Berger. "Direct characterization and
    removal of interfering absorption trends in two-layer turbid media."
    JOSA A 22.9 (2005): 1874-1882.

    Scholkmann, Felix, Andreas Jaakko Metz, and Martin Wolf.
    "Measuring tissue hemodynamics and oxygenation by continuous-wave
    functional near-infrared spectroscopy—how robust are the different
    calculation methods against movement artifacts?."
    Physiological measurement 35.4 (2014): 717.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    max_dist : number
        Channels less than this distance are considered short (m).

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    picks_od = pick_types(raw.info, fnirs='fnirs_od')

    if len(picks_od) == 0:
        raise RuntimeError('Data must be optical density.')

    distances = source_detector_distances(raw.info)

    picks_short = picks_od[distances[picks_od] < max_dist]
    picks_long = picks_od[distances[picks_od] > max_dist]

    if len(picks_short) == 0:
        raise RuntimeError('No short channels present.')
    if len(picks_long) == 0:
        raise RuntimeError('No long channels present.')

    for pick in picks_long:

        short_idx = _find_nearest_short(raw, pick, picks_short)

        A_l = raw.get_data(pick).ravel()
        A_s = raw.get_data(short_idx).ravel()

        # Eqn 27 Scholkmann et al 2014
        alfa = np.dot(A_s, A_l) / np.dot(A_s, A_s)

        # Eqn 26 Scholkmann et al 2014
        raw._data[pick] = A_l - alfa * A_s

    return raw


def _find_nearest_short(raw, pick, short_picks):
    """"
    Return index of closest short channel

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    pick : integer
        Index of channel of interest.
    short_picks : list of integers
        Indices of short channels.

    Returns
    -------
    pick : integer
        Index of short channel that is closest to pick.
        Note this index is relative to raw, and is from the subset provided
        in short_picks.

    """

    dist = [linalg.norm(raw.info['chs'][pick]['loc'][:3] -
                        raw.info['chs'][p_sh]['loc'][:3])
            for p_sh in short_picks]

    return short_picks[np.argmin(dist)]
