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
    Generate a design matrix based on annotations and model HRF.

    This is a wrapper function for the nilearn :footcite:`abraham2014machine`
    function ``make_first_level_design_matrix``. For detailed description
    of the arguments see the nilearn documentation at http://nilearn.github.io

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    stim_dur : Number
        The length of your stimulus.

    hrf_model : {'glover', 'spm', 'spm + derivative', \
        'spm + derivative + dispersion',\
        'glover + derivative', 'glover + derivative + dispersion',\
        'fir', None}, optional
        Specifies the hemodynamic response function. Default='glover'.

    drift_model : {'cosine', 'polynomial', None}, optional
        Specifies the desired drift model. Default='cosine'.

    high_pass : float, optional
        High-pass frequency in case of a cosine model (in Hz).
        Default=0.01.

    drift_order : int, optional
        Order of the drift model (in case it is polynomial).
        Default=1.

    fir_delays : array of shape(n_onsets) or list, optional
        In case of FIR design, yields the array of delays used in the FIR
        model (in scans). Default=[0].

    add_regs : array of shape(n_frames, n_add_reg) or pandas DataFrame
        Additional user-supplied regressors, e.g. data driven noise regressors
        or seed based regressors.

    add_reg_names : list of (n_add_reg,) str, optional
        If None, while add_regs was provided, these will be termed
        'reg_%i', i = 0..n_add_reg - 1
        If add_regs is a DataFrame, the corresponding column names are used
        and add_reg_names is ignored.

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    oversampling : int, optional
        Oversampling factor used in temporal convolutions. Default=50.

    Returns
    -------
    design_matrix : DataFrame instance,
        Holding the computed design matrix, the index being the frames_times
        and each column a regressor.

    References
    ----------
    .. footbibliography::
    """
    from nilearn.glm.first_level import make_first_level_design_matrix
    from pandas import DataFrame

    frame_times = raw.times

    # Create events for nilearn
    conditions = raw.annotations.description
    onsets = raw.annotations.onset - raw.first_time
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


def create_boxcar(raw, event_id=None, stim_dur=1):
    """
    Generate boxcar representation of the experimental paradigm.

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
        Returns an array for each annotation label.
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


def longest_inter_annotation_interval(raw):
    """
    Compute longest ISI per annotation.

    Specifically, longest period between two trials of
    the same condition.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    longest : list
        Longest ISI per annotation.
    annotation_name : list
        Annotation name corresponding to reported interval.
    """
    annotation_name = np.unique(raw.annotations.description)
    longest = []
    for desc in annotation_name:
        mask = raw.annotations.description == desc
        longest.append(np.max(np.diff(raw.annotations.onset[mask])))
    return longest, annotation_name


def drift_high_pass(raw):
    """
    Compute cosine drift regressor high pass cut off.

    Value computed according to Nilearn :footcite:`abraham2014machine`
    `suggestion <http://nilearn.github.io/auto_examples/04_glm_first
    _level/plot_first_level_details.html#changing-the-drift-model>`__.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    cutoff : number
        Suggested high pass cut off.

    References
    ----------
    .. footbibliography::
    """
    longest, annotation_name = longest_inter_annotation_interval(raw)
    max_isi = np.max(longest)
    return 1 / (2 * max_isi)
