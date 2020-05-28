# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from mne.preprocessing.nirs import source_detector_distances
from mne.utils import _validate_type
from mne.io import BaseRaw
import mne


def get_short_channels(raw, max_dist=0.01):
    """
    Return a raw object containing only channels with a source detector
    separation less than specified.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    max_dist : number
        Maximum distance of returned channel.

    Returns
    -------
    raw : instance of Raw
        Raw instance with only short channels.
    """

    short_chans = raw.copy().load_data()
    _validate_type(short_chans, BaseRaw, 'raw')

    picks = mne.pick_types(short_chans.info, meg=False, eeg=False, fnirs=True)
    if not len(picks):
        raise RuntimeError('Short channel extraction for NIRS signals only.')

    dists = source_detector_distances(short_chans.info, picks=picks)
    short_chans.pick(picks[dists < max_dist])

    return short_chans


def get_long_channels(raw, min_dist=0.01):
    """
    Return a raw object containing only channels with a source detector
    separation greater than specified.


    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    min_dist : number
        Minimum distance of returned channel.

    Returns
    -------
    raw : instance of Raw
        Raw instance with only short channels.
    """

    long_chans = raw.copy().load_data()
    _validate_type(long_chans, BaseRaw, 'raw')

    picks = mne.pick_types(long_chans.info, meg=False, eeg=False, fnirs=True)
    if not len(picks):
        raise RuntimeError('Short channel extraction for NIRS signals only.')

    dists = source_detector_distances(long_chans.info, picks=picks)
    long_chans.pick(picks[dists > min_dist])

    return long_chans
