# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from mne import Annotations, create_info
from mne.io import RawArray


def simulate_nirs_raw(sfreq=3., amplitude=1., annot_desc='A',
                      sig_dur=300., stim_dur=5.,
                      isi_min=15., isi_max=45.):
    """
    Create simulated data.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    sfreq : Number
        The sample rate.
    amplitude : Number, Array of numbers
        The amplitude of the signal to simulate in uM.
    annot_desc : String, Array of strings
        The name of the annotations for simulated amplitudes.
    sig_dur : Number
        The length of the signal to generate in seconds.
    stim_dur : Number, Array of numbers
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
    from nilearn.glm.first_level import make_first_level_design_matrix
    from pandas import DataFrame

    if type(amplitude) is not list:
        amplitude = [amplitude]
    if type(annot_desc) is not list:
        annot_desc = [annot_desc]
    if type(stim_dur) is not list:
        stim_dur = [stim_dur]

    frame_times = np.arange(sig_dur * sfreq) / sfreq

    assert len(amplitude) == len(annot_desc), "Same number of amplitudes as " \
                                              "annotations required."
    assert len(amplitude) == len(stim_dur), "Same number of amplitudes as " \
                                            "durations required."

    onset = 0.
    onsets = []
    conditions = []
    durations = []
    while onset < sig_dur - 60:
        c_idx = np.random.randint(0, len(amplitude))
        onset += np.random.uniform(isi_min, isi_max) + stim_dur[c_idx]
        onsets.append(onset)
        conditions.append(annot_desc[c_idx])
        durations.append(stim_dur[c_idx])

    events = DataFrame({'trial_type': conditions,
                        'onset': onsets,
                        'duration': durations})

    dm = make_first_level_design_matrix(frame_times, events,
                                        drift_model='polynomial',
                                        drift_order=0)
    dm = dm.drop(columns='constant')

    annotations = Annotations(onsets, durations, conditions)

    info = create_info(ch_names=['Simulated'], sfreq=sfreq, ch_types=['hbo'])

    for idx, annot in enumerate(annot_desc):
        if annot in dm.columns:
            dm[annot] *= amplitude[idx]

    a = np.sum(dm.to_numpy(), axis=1) * 1.e-6
    a = a.reshape(-1, 1).T

    raw = RawArray(a, info, verbose=False)
    raw.set_annotations(annotations)

    return raw
