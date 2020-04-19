"""
Experimental design and analysis.
"""

import numpy as np
import mne


def create_first_level_design_matrix(raw, stim_dur=1.,
                                     hrf_model='glover', drift_model='cosine',
                                     high_pass=0.01, drift_order=1,
                                     fir_delays=[0], add_regs=None,
                                     add_reg_names=None, min_onset=-24,
                                     oversampling=50):
    """
    Generate a design matrix for the experiment.

    This is a wrapper function for
    nilearn.stats.first_level_model.make_first_level_design_matrix.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    stim_dur : Number
        The length of your stimulus.
        TODO: Make this be independent per event id
    hrf_model : As specified in Nilearn
    drift_model : As specified in Nilearn
    high_pass : As specified in Nilearn. Actually its not documented there.
    drift_order : As specified in Nilearn
    fir_delays : As specified in Nilearn
    add_regs : As specified in Nilearn
    min_onset : As specified in Nilearn
    oversampling : As specified in Nilearn

    Returns
    -------
    dm : Design matrix
        As specified in Nilearn.
    """
    from nilearn.stats.first_level_model import make_first_level_design_matrix
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


def run_GLM(raw, design_matrix, noise_model='ar1', bins=100,
            n_jobs=1, verbose=0):
    """
    Run GLM on data using supplied design matrix.

    This is a wrapper function for nilearn.stats.first_level_model.run_glm.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    design_matrix : as specified in Nilearn
        Design matrix of experiment.
    noise_model : As specified in Nilearn
    bins : As specified in Nilearn
    n_jobs : As specified in Nilearn
    verbose : As specified in Nilearn

    Returns
    -------
    labels : array
        A map of values on voxels used to identify the corresponding model.
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    """
    from nilearn.stats.first_level_model import run_glm

    labels, glm_estimates = run_glm(raw._data.T, design_matrix.values,
                                    noise_model=noise_model, bins=bins,
                                    n_jobs=n_jobs, verbose=verbose)

    return labels, glm_estimates


def plot_GLM_topo(raw, labels, glm_estimates, design_matrix,
                  requested_conditions=None,
                  figsize=(12, 7), sphere=None):
    """
    Plot topomap of NIRS GLM data.

      .. warning:: Work in progress: I am trying to think on the best API.

      .. warning:: TODO: best method for plotting on specified components?

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    labels : array
        A map of values on voxels used to identify the corresponding model.
    requested_conditions : array
        Which conditions should be displayed.
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    design_matrix : DataFrame
        As specified in Nilearn
    figsize : TODO: Remove this, how does MNE ususally deal with this?
    sphere : As specified in MNE

    Returns
    -------
    fig : Figure of each design matrix componenent for hbo (top row)
          and hbr (bottom row).
    """

    import matplotlib.pyplot as plt

    estimates = np.zeros((len(labels), len(design_matrix.columns)))

    for idx, lab in enumerate(labels):
        matching_labels = np.where(([lab == l for l in labels]))
        matching_idx = np.where([idx == ml for ml in matching_labels])[1]
        estimates[idx, :] = glm_estimates[lab].theta[:, matching_idx].T

    types = np.unique(raw.get_channel_types())

    if requested_conditions is None:
        requested_conditions = design_matrix.columns

    fig, axes = plt.subplots(nrows=len(types),
                             ncols=len(requested_conditions),
                             figsize=figsize)

    estimates = estimates[:, [c in requested_conditions
                              for c in design_matrix.columns]]

    design_matrix = design_matrix[requested_conditions]

    vmax = np.max(np.abs(estimates))
    vmin = vmax * -1.

    for t_idx, t in enumerate(types):

        picks = mne.io.pick._picks_to_idx(raw.info, t, exclude=[],
                                          allow_empty=True)
        raw_subset = raw.copy().pick(picks=picks)

        _, pos, merge_channels, ch_names, ch_type, sphere, clip_origin = \
            mne.viz.topomap._prepare_topomap_plot(raw_subset, t, sphere=sphere)

        for idx, label in enumerate(design_matrix.columns):
            if label in requested_conditions:
                mne.viz.topomap.plot_topomap(estimates[picks, idx], pos,
                                             extrapolate='local',
                                             names=ch_names,
                                             vmin=vmin,
                                             vmax=vmax,
                                             axes=axes[t_idx, idx],
                                             show=False,
                                             sphere=sphere)
                axes[t_idx, idx].set_title(label)

    return fig


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
