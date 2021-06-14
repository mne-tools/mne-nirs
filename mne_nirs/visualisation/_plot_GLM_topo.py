# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import mne
from mne.utils import warn
from mne.channels.layout import _merge_ch_data
from mne.io.pick import _picks_to_idx


def plot_glm_topo(raw, glm_estimates, design_matrix,
                  requested_conditions=None,
                  axes=None, vmin=None, vmax=None, colorbar=True,
                  figsize=(12, 7), sphere=None):
    """
    Plot topomap of NIRS GLM data.

    If the channels in raw is a subset of those in the GLM estimate,
    then only the subset in raw will be plotted.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    design_matrix : DataFrame
        As specified in Nilearn.
    requested_conditions : array
        Which conditions should be displayed.
    axes : instance of Axes | None
        The axes to plot to. If None, a new figure is used.
    vmin : float | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        Defaults to None.
    vmax : float | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. Defaults to None.
    colorbar : Bool
        Should a colorbar be plotted.
    figsize : two values
        Figure size.
    sphere : As specified in MNE
        Sphere parameter from mne.viz.topomap.plot_topomap.

    Returns
    -------
    fig : figure
        Figure of each design matrix componenent for hbo (top row)
        and hbr (bottom row).
    """

    if not (raw.ch_names == list(glm_estimates.keys())):
        if len(raw.ch_names) < len(list(glm_estimates.keys())):
            warn("Reducing GLM results to match MNE data")
            glm_estimates = {a: glm_estimates[a] for a in raw.ch_names}
        else:
            raise RuntimeError('MNE data structure does not match regression '
                               f'results. Raw = {len(raw.ch_names)}. '
                               f'GLM = {len(list(glm_estimates.keys()))}')

    estimates = np.zeros((len(glm_estimates), len(design_matrix.columns)))

    for idx, name in enumerate(glm_estimates.keys()):
        estimates[idx, :] = glm_estimates[name].theta.T

    types = np.unique(raw.get_channel_types())

    if requested_conditions is None:
        requested_conditions = design_matrix.columns

    # Plotting setup
    if axes is None:
        fig, axes = plt.subplots(nrows=len(types),
                                 ncols=len(requested_conditions),
                                 figsize=figsize)

    estimates = estimates[:, [c in requested_conditions
                              for c in design_matrix.columns]]

    estimates = estimates * 1e6
    design_matrix = design_matrix[requested_conditions]

    if vmax is None:
        vmax = np.max(np.abs(estimates))
    if vmin is None:
        vmin = vmax * -1.
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for t_idx, t in enumerate(types):

        estmrg, pos, chs, sphere = _handle_overlaps(raw, t, sphere, estimates)

        for idx, label in enumerate(design_matrix.columns):
            if label in requested_conditions:

                # Deal with case when only a single
                # chroma or condition is available
                if (len(requested_conditions) == 1) & (len(types) == 1):
                    ax = axes
                elif (len(requested_conditions) == 1) & (len(types) > 1):
                    ax = axes[t_idx]
                elif (len(requested_conditions) > 1) & (len(types) == 1):
                    ax = axes[idx]
                else:
                    ax = axes[t_idx, idx]

                mne.viz.topomap.plot_topomap(estmrg[:, idx], pos,
                                             extrapolate='local',
                                             names=chs,
                                             vmin=vmin,
                                             vmax=vmax,
                                             cmap=cmap,
                                             axes=ax,
                                             show=False,
                                             sphere=sphere)
                ax.set_title(label)

        if colorbar:
            ax1_divider = make_axes_locatable(ax)
            cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                             orientation='vertical')
            cbar.set_label('Haemoglobin (uM)', rotation=270)

    return _get_fig_from_axes(axes)


def plot_glm_contrast_topo(raw, contrast, figsize=(12, 7), sphere=None):
    """
    Plot topomap of NIRS GLM data.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    contrast : dict
        As in nilearn.stats.compute_contrast.
    figsize : numbers
        TODO: Remove this, how does MNE usually deal with this.
    sphere : numbers
        As specified in MNE.

    Returns
    -------
    fig : figure
        Figure of each design matrix componenent for hbo (top row)
        and hbr (bottom row).
    """

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

        estmrg, pos, chs, sphere = _handle_overlaps(raw, t, sphere, estimates)

        # Deal with case when only a single chroma is available
        if len(types) == 1:
            ax = axes
        else:
            ax = axes[t_idx]

        # Plot the topomap
        mne.viz.topomap.plot_topomap(estmrg, pos,
                                     extrapolate='local',
                                     names=chs,
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
                        threshold=False,
                        vmin=None,
                        vmax=None,
                        cmap=None,
                        sensors=True,
                        res=64,
                        sphere=None,
                        colorbar=True,
                        show_names=False,
                        extrapolate='local',
                        image_interp='bilinear'):
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
    axes : instance of Axes | None
        The axes to plot to. If None, the current axes will be used.
    threshold : Bool
        If threshold is true, all values with P>|z| greater than 0.05 will
        be set to zero.
    vmin : float | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        Defaults to None.
    vmax : float | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. Defaults to None.
    cmap : matplotlib colormap | None
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True (default), circles
        will be used.
    res : int
        The resolution of the topomap image (n pixels along each side).
    sphere : numbers
        As specified in mne.
    colorbar : bool
        Should a colorbar be plotted.
    show_names : bool
        Display the channel names.
    extrapolate : str
        Type of extrapolation for image.
    image_interp : str
        Type of interpolation for image.

    Returns
    -------
    fig : figure
        Figure with topographic representation of statsmodel_df value.
    """

    # Check that the channels in two inputs match
    if not (raw.ch_names == list(statsmodel_df["ch_name"].values)):
        if len(raw.ch_names) < len(list(statsmodel_df["ch_name"].values)):
            print("Reducing GLM results to match MNE data")
            statsmodel_df["Keep"] = [g in raw.ch_names
                                     for g in statsmodel_df["ch_name"]]
            statsmodel_df = statsmodel_df.query("Keep == True")
        else:
            warn("MNE data structure does not match regression results")
    statsmodel_df = statsmodel_df.set_index('ch_name')
    statsmodel_df = statsmodel_df.reindex(raw.ch_names)

    # Extract estimate of interest to plot
    estimates = statsmodel_df[value].values

    if threshold:
        p = statsmodel_df["P>|z|"].values
        t = p > 0.05
        estimates[t] = 0.

    assert len(np.unique(statsmodel_df["Chroma"])) == 1,\
        "Only one Chroma allowed"

    if 'Condition' in statsmodel_df.columns:
        assert len(np.unique(statsmodel_df["Condition"])) == 1,\
            "Only one condition allowed"
        c = np.unique(statsmodel_df["Condition"])[0]
    else:
        c = "Contrast"

    t = np.unique(statsmodel_df["Chroma"])[0]

    # Plotting setup
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

    estmrg, pos, chs, sphere = _handle_overlaps(raw, t, sphere, estimates)

    mne.viz.topomap.plot_topomap(estmrg, pos,
                                 extrapolate=extrapolate,
                                 image_interp=image_interp,
                                 names=chs,
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=cmap,
                                 axes=axes,
                                 sensors=sensors,
                                 res=res,
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


def _handle_overlaps(raw, t, sphere, estimates):
    """Prepare for topomap including merging channels"""
    picks = _picks_to_idx(raw.info, t, exclude=[], allow_empty=True)
    raw_subset = raw.copy().pick(picks=picks)
    _, pos, merge_channels, ch_names, ch_type, sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(raw_subset, t, sphere=sphere)
    estmrg, ch_names = _merge_ch_data(estimates.copy()[picks], t, ch_names)
    return estmrg, pos, ch_names, sphere


def _get_fig_from_axes(ax):
    if issubclass(type(ax), mpl.axes.SubplotBase):
        return ax.figure
    elif type(ax) is np.ndarray:
        return _get_fig_from_axes(ax[0])
    else:
        raise RuntimeError(f"Unable to extract figure from {ax}")
