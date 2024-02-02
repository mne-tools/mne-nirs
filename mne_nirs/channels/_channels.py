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


def drop_sources(raw, sources):
    """
    Drop sources.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing fNIRS data.
    sources : iterable or str
        Iterable (e.g. list) of source(s) or source to remove.

    Returns
    -------
    inst : instance of Raw
        The modified instance.
    """
    if isinstance(sources, int):
        sources = [sources]

    try:
        all_str = all([isinstance(src, int) for src in sources])
    except TypeError:
        raise ValueError("'ch_names' must be iterable, got "
                         "type {} ({}).".format(type(sources), sources))

    if not all_str:
        raise ValueError("Each element in 'ch_names' must be int, got "
                         "{}.".format([type(ch) for ch in sources]))

    keeps = np.ones(len(raw.ch_names))
    for src in sources:
        template = f"S{src}_"
        for idx in range(len(raw.ch_names)):
            if template in raw.ch_names[idx]:
                keeps[idx] = 0

    keeps = np.where(keeps)[0]

    return raw.pick(keeps)


def drop_detectors(raw, detectors):
    """
    Drop detectors.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing fNIRS data.
    detectors : iterable or str
        Iterable (e.g. list) of detector(s) or detector to remove.

    Returns
    -------
    inst : instance of Raw
        The modified instance.
    """
    if isinstance(detectors, int):
        detectors = [detectors]

    try:
        all_str = all([isinstance(det, int) for det in detectors])
    except TypeError:
        raise ValueError("'ch_names' must be iterable, got "
                         "type {} ({}).".format(type(detectors), detectors))

    if not all_str:
        raise ValueError("Each element in 'ch_names' must be int, got "
                         "{}.".format([type(det) for det in detectors]))

    keeps = np.ones(len(raw.ch_names))
    for det in detectors:
        template = f"_D{det} "
        for idx in range(len(raw.ch_names)):
            if template in raw.ch_names[idx]:
                keeps[idx] = 0

    keeps = np.where(keeps)[0]

    return raw.pick(keeps)


def pick_sources(raw, sources):
    """
    Pick sources.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing fNIRS data.
    sources : iterable or str
        Iterable (e.g. list) of source(s) or source to keep.

    Returns
    -------
    inst : instance of Raw
        The modified instance.
    """
    if isinstance(sources, int):
        sources = [sources]

    try:
        all_str = all([isinstance(src, int) for src in sources])
    except TypeError:
        raise ValueError("'ch_names' must be iterable, got "
                         "type {} ({}).".format(type(sources), sources))

    if not all_str:
        raise ValueError("Each element in 'ch_names' must be int, got "
                         "{}.".format([type(ch) for ch in sources]))

    keeps = np.zeros(len(raw.ch_names))
    for src in sources:
        template = f"S{src}_"
        for idx in range(len(raw.ch_names)):
            if template in raw.ch_names[idx]:
                keeps[idx] = 1

    keeps = np.where(keeps)[0]

    return raw.pick(keeps)


def pick_detectors(raw, detectors):
    """
    Pick detectors.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing fNIRS data.
    detectors : iterable or str
        Iterable (e.g. list) of detector(s) or detector to keep.

    Returns
    -------
    inst : instance of Raw
        The modified instance.
    """
    if isinstance(detectors, int):
        detectors = [detectors]

    try:
        all_str = all([isinstance(det, int) for det in detectors])
    except TypeError:
        raise ValueError("'ch_names' must be iterable, got "
                         "type {} ({}).".format(type(detectors), detectors))

    if not all_str:
        raise ValueError("Each element in 'ch_names' must be int, got "
                         "{}.".format([type(det) for det in detectors]))

    keeps = np.zeros(len(raw.ch_names))
    for det in detectors:
        template = f"_D{det} "
        for idx in range(len(raw.ch_names)):
            if template in raw.ch_names[idx]:
                keeps[idx] = 1

    keeps = np.where(keeps)[0]

    return raw.pick(keeps)
