# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np

from mne.io import RawArray
from mne import pick_types
from mne.utils import _validate_type
from mne import Annotations, create_info


def simulate_nirs_data(fs=3., signal_length_s=300.,
                       isi_min=15., isi_max=45., duration=5.,
                       amplitude=3.):
    """
    Create simulated data.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.

    events : DataFrame
        Dataframe containing event information.
    """
    from nilearn.stats.first_level_model import make_first_level_design_matrix
    from pandas import DataFrame

    frame_times = np.arange(signal_length_s * fs) / fs

    onset = 0.
    onsets = []
    conditions = []
    durations = []
    while onset < signal_length_s - 60:
        onset += np.random.uniform(isi_min, isi_max)
        onsets.append(onset)
        conditions.append("A")
        durations.append(duration)

    events = DataFrame({'trial_type': conditions,
                        'onset': onsets,
                        'duration': duration})

    dm = make_first_level_design_matrix(frame_times, events,
                                        drift_model='polynomial',
                                        drift_order=0)

    annotations = Annotations(onsets, durations, conditions)

    info = create_info(ch_names=['Simulated'], sfreq=fs, ch_types=['hbo'])

    raw = RawArray(dm[["A"]].to_numpy().T * amplitude * 1.e-6, info)
    raw.set_annotations(annotations)

    return raw
