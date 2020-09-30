# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import numpy as np
import mne
from mne.utils import warn
from mne.channels.layout import _merge_ch_data


def plot_glm_topo(raw, glm_estimates, design_matrix,
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

        if sum(["x" in ch for ch in ch_names]):
            warn("Channels were merged")
            keeps = np.array(np.where(["x" not in ch for ch in ch_names])[0])
            picks = picks[keeps]

        for idx, label in enumerate(design_matrix.columns):
            if label in requested_conditions:
                mne.viz.topomap.plot_topomap(estimates[picks, idx], pos,
                                             extrapolate='local',
                                             names=ch_names,
                                             vmin=vmin,
                                             vmax=vmax,
                                             cmap=cmap,
                                             axes=axes[t_idx, idx],
                                             show=False,
                                             sphere=sphere)
                axes[t_idx, idx].set_title(label)

        ax1_divider = make_axes_locatable(axes[t_idx, -1])
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                         orientation='vertical')
        cbar.set_label('Haemoglobin (uM)', rotation=270)

    return fig


def plot_glm_contrast_topo(raw, contrast,
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

    # Extract types. One subplot is created per type (hbo/hbr)
    types = np.unique(raw.get_channel_types())

    # Extract values to plot and rescale to uM
    estimates = contrast.effect[0]
    estimates = estimates * 1e6

    # Create subplots for figures
    fig, axes = plt.subplots(nrows=1,
                             ncols=len(types),
                             figsize=figsize)
    # Create limits for colorbar
    vmax = np.max(np.abs(estimates))
    vmin = vmax * -1.
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for t_idx, t in enumerate(types):
        # Extract just the channels corresponding to the type to plot
        picks = mne.io.pick._picks_to_idx(raw.info, t, exclude=[],
                                          allow_empty=True)
        raw_subset = raw.copy().pick(picks=picks)

        # Extract positions of channels for plotting
        _, pos, merge_channels, ch_names, ch_type, sphere, clip_origin = \
            mne.viz.topomap._prepare_topomap_plot(raw_subset, t, sphere=sphere)

        # Deal with case when only a single chroma is available
        if len(types) == 1:
            ax = axes
        else:
            ax = axes[t_idx]

        # Plot the topomap
        mne.viz.topomap.plot_topomap(estimates[picks], pos,
                                     extrapolate='local',
                                     names=ch_names,
                                     vmin=vmin,
                                     vmax=vmax,
                                     cmap=cmap,
                                     axes=ax,
                                     show=False,
                                     sphere=sphere)
        # Sets axes title
        if t == 'hbo':
            ax.set_title('Oxyhaemoglobin')
        elif t == 'hbr':
            ax.set_title('Deoxyhaemoglobin')
        else:
            ax.set_title(t)

    # Create a single colorbar for all types based on limits above
    ax1_divider = make_axes_locatable(ax)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                     orientation='vertical')
    cbar.set_label('Contrast Effect', rotation=270)

    return fig


def plot_glm_group_topo(raw, statsmodel_df,
                        value="Coef.",
                        axes=None,
                        sphere=None,
                        colorbar=True,
                        cmap=None,
                        threshold=False,
                        show_names=False,
                        extrapolate='local',
                        image_interp='bilinear',
                        vmin=None, vmax=None):
    """
    Plot topomap of NIRS group level GLM results.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    statsmodel_df : DataFrame
        Dataframe created from a statsmodel summary.
    value : String
        Which column in the `statsmodel_df` to use in the topo map.

    Returns
    -------
    fig : Figure with topographic representation of statsmodel_df value.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    if not (raw.ch_names == list(statsmodel_df["ch_name"].values)):
        if len(raw.ch_names) < len(list(statsmodel_df["ch_name"].values)):
            print("reducing GLM results to match raw")
            statsmodel_df["Keep"] = [g in raw.ch_names
                                     for g in statsmodel_df["ch_name"]]
            statsmodel_df = statsmodel_df.query("Keep == True")
        else:
            warn("MNE data structure does not match regression results")

    estimates = statsmodel_df[value].values

    if threshold:
        p = statsmodel_df["P>|z|"].values
        t = p > 0.05
        estimates[t] = 0.

    assert len(np.unique(statsmodel_df["Chroma"])) == 1,\
        "Only one Chroma allowed"

    if 'condition' in statsmodel_df.columns:
        assert len(np.unique(statsmodel_df["condition"])) == 1,\
            "Only one condition allowed"
        c = np.unique(statsmodel_df["condition"])[0]
    else:
        c = "Contrast"

    t = np.unique(statsmodel_df["Chroma"])

    if axes is None:
        fig, axes = plt.subplots(nrows=1,
                                 ncols=1,
                                 figsize=(12, 7))

    # Set limits of topomap and colors
    if vmax is None:
        vmax = np.max(np.abs(estimates))
    if vmin is None:
        vmin = vmax * -1.
    if cmap is None:
        cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Handle overlapping channels, these distort the topomap so are averaged.
    raw_subset = raw.copy()
    _, pos, merge_channels, ch_names, ch_type, sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(raw_subset, t, sphere=sphere)
    estimates, ch_names = _merge_ch_data(estimates, t, ch_names)

    if sum(["x" in ch for ch in ch_names]):
        warn("Channels were merged")
        # keeps = np.array(np.where(["x" not in ch for ch in ch_names])[0])
        # picks = picks[keeps]

    mne.viz.topomap.plot_topomap(estimates, pos,
                                 extrapolate=extrapolate,
                                 image_interp=image_interp,
                                 names=ch_names,
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=cmap,
                                 axes=axes,
                                 show=False,
                                 show_names=show_names,
                                 sphere=sphere)
    axes.set_title(c)

    if colorbar:
        ax1_divider = make_axes_locatable(axes)
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                         orientation='vertical')
        cbar.set_label(value, rotation=270)

    return axes
