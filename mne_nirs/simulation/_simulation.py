# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from mne import Annotations, create_info
from mne.io import RawArray


def simulate_block_design(sfreq=3., amplitude=1.,
                          sig_dur=300., stim_dur=5.,
                          isi_min=15., isi_max=45.):
    """
    Create simulated data.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    sfreq : Number
        The sample rate.
    amplitude : Number
        The amplitude of the signal to simulate in uM.
    sig_dur : Number
        The length of the signal to generate in seconds.
    stim_dur : Number
        The length of the stimulus to generate in seconds.
    isi_min : Number
        The minimum duration of the inter stimulus interval in seconds.
    isi_max : Number
        The maximum duration of the inter stimulus interval in seconds.

    Returns
    -------
    raw : instance of Raw
        The generated raw instance.
    """
    from nilearn.stats.first_level_model import make_first_level_design_matrix
    from pandas import DataFrame

    frame_times = np.arange(sig_dur * sfreq) / sfreq

    onset = 0.
    onsets = []
    conditions = []
    durations = []
    while onset < sig_dur - 60:
        onset += np.random.uniform(isi_min, isi_max) + stim_dur
        onsets.append(onset)
        conditions.append("A")
        durations.append(stim_dur)

    events = DataFrame({'trial_type': conditions,
                        'onset': onsets,
                        'duration': durations})

    dm = make_first_level_design_matrix(frame_times, events,
                                        drift_model='polynomial',
                                        drift_order=0)

    annotations = Annotations(onsets, durations, conditions)

    info = create_info(ch_names=['Simulated'], sfreq=sfreq, ch_types=['hbo'])

    raw = RawArray(dm[["A"]].to_numpy().T * amplitude * 1.e-6,
                   info, verbose=False)
    raw.set_annotations(annotations)

    return raw
