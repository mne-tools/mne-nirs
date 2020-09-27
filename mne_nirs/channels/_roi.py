# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import numpy as np
from mne.utils import warn


def picks_pair_to_idx(raw, sd_pairs, on_missing='error'):
    """
    Return a list of picks for specified source detector pairs.

    If multiple channel have the same source detector pair,
    for example if there are multiple wavelengths or chromaphore, then
    all channels matching the requested sources and detectors will be
    returned.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    sd_pairs : list of lists
        List of source detector pairs. For example, to request the picks for
        channels comprising of source 1 detector 4 and source 13
        detector 4 you would specify [[1, 4], [13, 4]].
    on_missing : str
        What to do if one or several requested source detector are not found
        in the recording.
        Valid keys are 'error' | 'warning' | 'ignore'
        Default is 'error'. If on_missing is 'warning' it will proceed but
        warn, if 'ignore' it will proceed silently. Note.
        If none of the event ids are found in the data, an error will be
        automatically generated irrespective of this parameter.

    Returns
    -------
    picks : list of integers
        List of picks corresponding to requested source detector pairs.
    """

    ch_names = raw.ch_names
    picks = list()

    for pair in sd_pairs:
        pair_name = "S" + str(pair[0]) + "_D" + str(pair[1]) + " "
        pair_picks = np.where([pair_name in ch for ch in ch_names])[0]
        if len(pair_picks) == 0:
            msg = ('No matching channels found for source %s '
                   'detector %s' % (pair[0], pair[1]))
            if on_missing == 'error':
                print(pair_picks)
                raise ValueError(msg)
            elif on_missing == 'warning':
                warn(msg)
            else:
                # on_missing == 'ignore':
                continue
        [picks.append(pick) for pick in pair_picks]

    return picks
