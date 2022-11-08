# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from copy import deepcopy
import inspect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from mne import Info
from mne.utils import warn
from mne.channels.layout import _merge_ch_data
from mne.io.pick import _picks_to_idx, _get_channel_types, pick_info
from mne.viz import plot_topomap


def _plot_glm_topo(inst, glm_estimates, design_matrix, *,
                   requested_conditions=None,
                   axes=None, vlim=None, vmin=None, vmax=None, colorbar=True,
                   figsize=(12, 7), sphere=None):

    info = deepcopy(inst if isinstance(inst, Info) else inst.info)

    if not (info.ch_names == list(glm_estimates.keys())):
        if len(info.ch_names) < len(list(glm_estimates.keys())):
            warn("Reducing GLM results to match MNE data")
            glm_estimates = {a: glm_estimates[a] for a in info.ch_names}
        else:
            raise RuntimeError('MNE data structure does not match regression '
                               f'results. Raw = {len(info.ch_names)}. '
                               f'GLM = {len(list(glm_estimates.keys()))}')

    estimates = np.zeros((len(glm_estimates), len(design_matrix.columns)))

    for idx, name in enumerate(glm_estimates.keys()):
        estimates[idx, :] = glm_estimates[name].theta.T

    types = np.unique(_get_channel_types(info))

    if requested_conditions is None:
        requested_conditions = design_matrix.columns
    requested_conditions = [x for x in design_matrix.columns
                            if x in requested_conditions]

    # Plotting setup
    if axes is None:
        fig, axes = plt.subplots(nrows=len(types),
                                 ncols=len(requested_conditions),
                                 figsize=figsize)

    estimates = estimates[:, [c in requested_conditions
                              for c in design_matrix.columns]]

    estimates = estimates * 1e6
    design_matrix = design_matrix[requested_conditions]
    vlim, vlim_kwargs = _handle_vlim(vlim, vmin, vmax, estimates)
    del vmin, vmax
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

    for t_idx, t in enumerate(types):

        estmrg, pos, chs, sphere = _handle_overlaps(info, t, sphere, estimates)

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

                plot_topomap(
                    estmrg[:, idx], pos, extrapolate='local', names=chs,
                    cmap=cmap, axes=ax, show=False, sphere=sphere,
                    **vlim_kwargs)
                ax.set_title(label)

        if colorbar:
            ax1_divider = make_axes_locatable(ax)
            cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                             orientation='vertical')
            cbar.set_label('Haemoglobin (uM)', rotation=270)

    return _get_fig_from_axes(axes)


def _plot_glm_contrast_topo(inst, contrast, figsize=(12, 7), sphere=None):

    info = deepcopy(inst if isinstance(inst, Info) else inst.info)

    # Extract types. One subplot is created per type (hbo/hbr)
    types = np.unique(_get_channel_types(info))

    # Extract values to plot and rescale to uM
    estimates = contrast.effect[0]
    estimates = estimates * 1e6

    # Create subplots for figures
    fig, axes = plt.subplots(nrows=1,
                             ncols=len(types),
                             figsize=figsize)
    # Create limits for colorbar
    vlim, vlim_kwargs = _handle_vlim((None, None), None, None, estimates)
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

    for t_idx, t in enumerate(types):

        estmrg, pos, chs, sphere = _handle_overlaps(info, t, sphere, estimates)

        # Deal with case when only a single chroma is available
        if len(types) == 1:
            ax = axes
        else:
            ax = axes[t_idx]

        # Plot the topomap
        plot_topomap(
            estmrg, pos, extrapolate='local', names=chs, cmap=cmap, axes=ax,
            show=False, sphere=sphere, **vlim_kwargs)
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


def plot_glm_group_topo(inst, statsmodel_df,
                        value="Coef.",
                        axes=None,
                        threshold=False,
                        *,
                        vlim=(None, None),
                        vmin=None,
                        vmax=None,
                        cmap=None,
                        sensors=True,
                        res=64,
                        sphere=None,
                        colorbar=True,
                        names=False,
                        show_names=None,
                        extrapolate='local',
                        image_interp='cubic'):
    """
    Plot topomap of NIRS group level GLM results.

    Parameters
    ----------
    inst : instance of Info or Raw
        Raw data or info structure used to generate the GLM results.
    statsmodel_df : DataFrame
        Dataframe created from a statsmodel summary.
    value : String
        Which column in the `statsmodel_df` to use in the topo map.
    axes : instance of Axes | None
        The axes to plot to. If None, the current axes will be used.
    threshold : Bool
        If threshold is true, all values with P>|z| greater than 0.05 will
        be set to zero.
    vlim : tuple of length 2
        Colormap limits to use. If a :class:`tuple` of floats, specifies the
        lower and upper bounds of the colormap (in that order); providing
        ``None`` for either entry will set the corresponding boundary at the
        min/max of the data (separately for each topomap).
    vmin : float | None
        Deprecated, use 'vlim' instead.
    vmax : float | None
        Deprecated, use 'vlim' instead.
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
    names : list of str
        The channel names to display.
    show_names : bool
        Deprecated, use ``names`` instead.
    extrapolate : str
        Type of extrapolation for image.
    image_interp : str
        Type of interpolation for image.

    Returns
    -------
    fig : figure
        Figure with topographic representation of statsmodel_df value.
    """
    info = deepcopy(inst if isinstance(inst, Info) else inst.info)
    if show_names is not None:
        names = show_names
        warn('show_names is deprecated and will be removed in the next '
             'release, use names instead', FutureWarning)
    del show_names

    # Check that the channels in two inputs match
    if not (info.ch_names == list(statsmodel_df["ch_name"].values)):
        if len(info.ch_names) < len(list(statsmodel_df["ch_name"].values)):
            print("Reducing GLM results to match MNE data")
            statsmodel_df["Keep"] = [g in info.ch_names
                                     for g in statsmodel_df["ch_name"]]
            statsmodel_df = statsmodel_df.query("Keep == True")
        else:
            warn("MNE data structure does not match regression results")
    statsmodel_df = statsmodel_df.set_index('ch_name')
    statsmodel_df = statsmodel_df.reindex(info.ch_names)

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
    vlim, vlim_kwargs = _handle_vlim(vlim, vmin, vmax, estimates)
    del vmin, vmax
    if cmap is None:
        cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

    estmrg, pos, chs, sphere = _handle_overlaps(info, t, sphere, estimates)
    if 'names' in inspect.signature(plot_topomap).parameters:
        names_kwarg = dict(names=chs if names else [''] * len(chs))
    else:
        names_kwarg = dict(show_names=names, names=chs)

    plot_topomap(
        estmrg, pos, extrapolate=extrapolate, image_interp=image_interp,
        cmap=cmap, axes=axes, sensors=sensors, res=res, show=False,
        sphere=sphere, **vlim_kwargs, **names_kwarg)
    axes.set_title(c)

    if colorbar:
        ax1_divider = make_axes_locatable(axes)
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm,
                                         orientation='vertical')
        cbar.set_label(value, rotation=270)

    return axes


def _handle_overlaps(info, t, sphere, estimates):
    """Prepare for topomap including merging channels"""
    from mne.viz.topomap import _prepare_topomap_plot
    picks = _picks_to_idx(info, t, exclude=[], allow_empty=True)
    info_subset = pick_info(info, picks)
    _, pos, merge_channels, ch_names, ch_type, sphere, clip_origin = \
        _prepare_topomap_plot(info_subset, t, sphere=sphere)
    estmrg, ch_names = _merge_ch_data(estimates.copy()[picks], t, ch_names)
    return estmrg, pos, ch_names, sphere


def _get_fig_from_axes(ax):
    if isinstance(ax, mpl.axes.SubplotBase):
        return ax.figure
    elif type(ax) is np.ndarray:
        return _get_fig_from_axes(ax[0])
    else:
        raise RuntimeError(f"Unable to extract figure from {ax}")


def _handle_vlim(vlim, vmin, vmax, estimates):
    if vmin is not None or vmax is not None:
        warn('vmin and vmax are deprecated and will be removed in the next '
             'release, please use vlim instead', FutureWarning)
        vlim = (vmin, vmax)
    else:
        vmin, vmax = vlim
    if vmax is None:
        vmax = np.max(np.abs(estimates))
    if vmin is None:
        vmin = vmax * -1.
    vlim = tuple(vlim)
    if 'vlim' in inspect.signature(plot_topomap).parameters:
        kwargs = dict(vlim=(vmin, vmax))
    else:
        kwargs = dict(vmin=vmin, vmax=vmax)
    return vlim, kwargs
