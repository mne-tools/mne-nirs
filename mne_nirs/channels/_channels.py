# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
import re
from mne import pick_types
from mne.utils import _validate_type
from mne.io import BaseRaw


def list_sources(raw):
    """
    List all the sources in the fNIRS montage.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing fNIRS data.

    Returns
    -------
    sources : list
        Unique list of all sources in ascending order.
    """
    _validate_type(raw, BaseRaw, 'raw')

    picks = pick_types(raw.info, meg=False, eeg=False, fnirs=True,
                       exclude=[])
    if not len(picks):
        raise RuntimeError('Listing source is for fNIRS signals only.')

    sources = list()
    ch_names = raw.ch_names
    for pick in picks:
        x = re.search(r"S(\d+)_D(\d+)", ch_names[pick])
        sources.append(int(x.group(1)))

    sources = np.unique(sources)

    return sources


def list_detectors(raw):
    """
    List all the detectors in the fNIRS montage.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing fNIRS data.

    Returns
    -------
    sources : list
        Unique list of all detectors in ascending order.
    """
    _validate_type(raw, BaseRaw, 'raw')

    picks = pick_types(raw.info, meg=False, eeg=False, fnirs=True,
                       exclude=[])
    if not len(picks):
        raise RuntimeError('Listing source is for fNIRS signals only.')

    detectors = list()
    ch_names = raw.ch_names
    for pick in picks:
        x = re.search(r"S(\d+)_D(\d+)", ch_names[pick])
        detectors.append(int(x.group(2)))

    detectors = np.unique(detectors)

    return detectors
