# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import numpy as np
import mne
from mne.utils import warn


def plot_GLM_topo(raw, glm_estimates, design_matrix,
                  requested_conditions=None,
                  figsize=(12, 7), sphere=None):
    """
    Plot topomap of NIRS GLM data.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
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
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    if not (raw.ch_names == list(glm_estimates.keys())):
        warn("MNE data structure does not match regression results")

    estimates = np.zeros((len(glm_estimates), len(design_matrix.columns)))

    for idx, name in enumerate(glm_estimates.keys()):
        estimates[idx, :] = glm_estimates[name].theta.T

    types = np.unique(raw.get_channel_types())

    if requested_conditions is None:
        requested_conditions = design_matrix.columns

    fig, axes = plt.subplots(nrows=len(types),
                             ncols=len(requested_conditions),
                             figsize=figsize)

    estimates = estimates[:, [c in requested_conditions
                              for c in design_matrix.columns]]

    estimates = estimates * 1e6
    design_matrix = design_matrix[requested_conditions]

    vmax = np.max(np.abs(estimates))
    vmin = vmax * -1.
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

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

        ax1_divider = make_axes_locatable(axes[t_idx, -1])
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                         orientation='vertical')
        cbar.set_label('Haemoglobin (uV)', rotation=270)

    return fig


def plot_GLM_contrast_topo(raw, contrast,
                           figsize=(12, 7), sphere=None):
    """
    Plot topomap of NIRS GLM data.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    contrast : dict
        nilearn.stats.compute_contrast
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

    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    types = np.unique(raw.get_channel_types())

    fig, axes = plt.subplots(nrows=1,
                             ncols=len(types),
                             figsize=figsize)

    estimates = contrast.effect[0]
    estimates = estimates * 1e6

    vmax = np.max(np.abs(estimates))
    vmin = vmax * -1.
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for t_idx, t in enumerate(types):
        picks = mne.io.pick._picks_to_idx(raw.info, t, exclude=[],
                                          allow_empty=True)
        raw_subset = raw.copy().pick(picks=picks)

        _, pos, merge_channels, ch_names, ch_type, sphere, clip_origin = \
            mne.viz.topomap._prepare_topomap_plot(raw_subset, t, sphere=sphere)

        mne.viz.topomap.plot_topomap(estimates[picks], pos,
                                     extrapolate='local',
                                     names=ch_names,
                                     vmin=vmin,
                                     vmax=vmax,
                                     axes=axes[t_idx],
                                     show=False,
                                     sphere=sphere)
        axes[t_idx].set_title(t)

    ax1_divider = make_axes_locatable(axes[-1])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                     orientation='vertical')
    cbar.set_label('Haemoglobin (uV)', rotation=270)

    return fig
