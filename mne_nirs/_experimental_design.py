"""
This is a module to be used as a reference for building other modules
"""

import numpy as np
import mne


def foo():
    """
    Just a test.

    :return: str
        Return foo
    """
    print(2 * 2)
    return 'foo'


def create_boxcar(raw, event_id=None, stim_dur=5):
    """
    Create a boxcar of the experiment.

    **NOTE:** This is just a holder function while I figure out
    how to get github actions working.

    :return: array
        Returns an array for each trigger channel.
    """
    from scipy import signal
    bc = signal.boxcar(round(raw.info['sfreq'] * stim_dur))
    events, ids = mne.events_from_annotations(raw, event_id=event_id)
    s = np.zeros((len(raw.times), len(ids)))
    for idx, id in enumerate(ids):
        id_idx = [e[2] == idx + 1 for e in events]
        id_evt = events[id_idx]
        event_samples = [e[0] for e in id_evt]
        s[event_samples, idx] = 1.
        s[:, idx] = np.convolve(s[:, idx], bc)[:len(raw.times)]
    return s



def create_hrf(raw, event_id=None, stim_dur=5):
    """
    Calculate expected haemodynamic response function.

    **NOTE:** This is just a holder function while I figure out
    how to get github actions working.

    :return: array
        Returns an array for each trigger channel.
    """
    from bids.analysis import hrf
    h0 = hrf._gamma_difference_hrf(raw.info['sfreq'], time_length=32, delay=4)

    s = create_boxcar(raw, event_id=event_id, stim_dur=stim_dur)
    for idx in range(s.shape[1]):
        s[:, idx] = np.convolve(s[:, idx], h0)[:len(raw.times)]

    return s
