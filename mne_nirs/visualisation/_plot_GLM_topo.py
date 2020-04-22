# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import numpy as np
import mne


def plot_GLM_topo(raw, labels, glm_estimates, design_matrix,
                  requested_conditions=None,
                  figsize=(12, 7), sphere=None):
    """
    Plot topomap of NIRS GLM data.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    labels : array
        A map of values on voxels used to identify the corresponding model.
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    design_matrix : DataFrame
        As specified in Nilearn
    requested_conditions : array
        Which conditions should be displayed.
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
