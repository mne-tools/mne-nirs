# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
import mne


def make_first_level_design_matrix(raw, stim_dur=1.,
                                   hrf_model='glover',
                                   drift_model='cosine',
                                   high_pass=0.01, drift_order=1,
                                   fir_delays=[0], add_regs=None,
                                   add_reg_names=None, min_onset=-24,
                                   oversampling=50):
    """
    Generate a design matrix for the experiment.

    This is a wrapper function for
    nilearn.stats.first_level_model.make_first_level_design_matrix.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    stim_dur : Number
        The length of your stimulus.
    hrf_model : {'spm', 'spm + derivative', 'spm + derivative + dispersion',
        'glover', 'glover + derivative', 'glover + derivative + dispersion',
        'fir', None}, optional,
        Specifies the hemodynamic response function
    drift_model : {'polynomial', 'cosine', None}, optional
        Specifies the desired drift model,
    period_cut : float, optional
        Cut period of the high-pass filter in seconds.
        Used only if drift_model is 'cosine'.
    drift_order : int, optional
        Order of the drift model (in case it is polynomial).
    fir_delays : array of shape(n_onsets) or list, optional,
        In case of FIR design, yields the array of delays used in the FIR
        model (in scans).
    add_regs : array of shape(n_frames, n_add_reg), optional
        additional user-supplied regressors, e.g. data driven noise regressors
        or seed based regressors.
    add_reg_names : list of (n_add_reg,) strings, optional
        If None, while add_regs was provided, these will be termed
        'reg_%i', i = 0..n_add_reg - 1
    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
    oversampling: int, optional,
        Oversampling factor used in temporal convolutions.
    Returns
    -------
    design_matrix : DataFrame instance,
        holding the computed design matrix, the index being the frames_times
        and each column a regressor.
    oversampling : As specified in Nilearn

    Returns
    -------
    dm : Design matrix
        As specified in Nilearn.
    """
    from nilearn.glm.first_level import make_first_level_design_matrix
    from pandas import DataFrame

    frame_times = raw.times

    # Create events for nilearn
    conditions = raw.annotations.description
    onsets = raw.annotations.onset
    duration = stim_dur * np.ones(len(conditions))
    events = DataFrame({'trial_type': conditions,
                        'onset': onsets,
                        'duration': duration})

    dm = make_first_level_design_matrix(frame_times, events,
                                        drift_model=drift_model,
                                        drift_order=drift_order,
                                        hrf_model=hrf_model,
                                        min_onset=min_onset,
                                        high_pass=high_pass,
                                        add_regs=add_regs,
                                        oversampling=oversampling,
                                        add_reg_names=add_reg_names,
                                        fir_delays=fir_delays)

    return dm


def create_boxcar(raw, event_id=None, stim_dur=5):
    """
    Create a boxcar of the experiment.

      .. warning:: The naming of this function may change. Use with caution.
                   This is just a place holder while I get the documentation\
                   up and running.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    event_id : as specified in MNE
        Information about events.
    stim_dur : Number
        The length of your stimulus.

    Returns
    -------
    s : array
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
