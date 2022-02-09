# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import numpy as np
from copy import deepcopy

from mne import stc_near_sensors, EvokedArray, read_source_spaces, Info
from mne.io.constants import FIFF
from mne.utils import verbose, get_subjects_dir


@verbose
def plot_glm_surface_projection(inst, statsmodel_df, picks="hbo",
                                value="Coef.",
                                background='w', figure=None, clim='auto',
                                mode='weighted', colormap='RdBu_r',
                                surface='pial', hemi='both', size=800,
                                view=None, colorbar=True, distance=0.03,
                                subjects_dir=None, src=None, verbose=False):
    """
    Project GLM results on to the surface of the brain.

    Note: This function provides a convenient wrapper around low level
    MNE-Python functions. It is convenient if you wish to use a generic head
    model. If you have acquired fMRI images you may wish to use the underlying
    lower level functions.

    Note: This function does not conduct a forward model analysis with photon
    migration etc. It simply projects the values from each channel to the
    local cortical surface. It is useful for visualisation, but users should
    take this in to consideration when drawing conclusions from the
    visualisation.

    Parameters
    ----------
    inst : instance of Raw
        Haemoglobin data.
    statsmodel_df : dataframe
        As produced by produced by `statsmodels_to_results`.
    %(picks_base)s good sEEG, ECoG, and DBS channels.
    value : str
        Column from dataframe to plot.
    background : matplotlib color
        Color of the background of the display window.
    figure : mayavi.core.api.Scene, matplotlib.figure.Figure, list, int, None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the Mayavi
        figure by it's id or create a new figure with the given id. If an
        instance of matplotlib figure, mpl backend is used for plotting.
    %(clim)s
    mode : str
        Can be "sum" to do a linear sum of weights, "weighted" to make this
        a weighted sum, "nearest" to
        use only the weight of the nearest sensor, or "single" to
        do a distance-weight of the nearest sensor. Default is "sum".
    colormap : str
        Colormap to use.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str
        Hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    size : float or tuple of float
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
        Has no effect with mpl backend.
    view : str
        View to set brain to.
    colorbar : bool
        If True, display colorbar on scene.
    distance : float
        Distance (m) defining the activation "ball" of the sensor.
    %(subjects_dir)s
    src : instance of SourceSpaces
        The source space.
    %(verbose)s

    Returns
    -------
    figure : instance of mne.viz.Brain | matplotlib.figure.Figure
        An instance of :class:`mne.viz.Brain` or matplotlib figure.
    """
    info = deepcopy(inst if isinstance(inst, Info) else inst.info)
    if not (info.ch_names == list(statsmodel_df['ch_name'].values)):
        raise RuntimeError('MNE data structure does not match dataframe '
                           f'results.\nMNE = {info.ch_names}.\n'
                           f'GLM = {list(statsmodel_df["ch_name"].values)}')

    ea = EvokedArray(np.tile(statsmodel_df[value].values.T, (1, 1)).T,
                     info.copy())

    return _plot_3d_evoked_array(inst, ea, picks=picks,
                                 value=value,
                                 background=background, figure=figure,
                                 clim=clim,
                                 mode=mode, colormap=colormap,
                                 surface=surface, hemi=hemi, size=size,
                                 view=view, colorbar=colorbar,
                                 distance=distance,
                                 subjects_dir=subjects_dir, src=src,
                                 verbose=verbose)


def _plot_3d_evoked_array(inst, ea, picks="hbo",
                          value="Coef.",
                          background='w', figure=None, clim='auto',
                          mode='weighted', colormap='RdBu_r',
                          surface='pial', hemi='both', size=800,
                          view=None, colorbar=True, distance=0.03,
                          subjects_dir=None, src=None, verbose=False):

    # TODO: mimic behaviour of other MNE-NIRS glm plotting options
    if picks is not None:
        ea = ea.pick(picks=picks)

    if subjects_dir is None:
        subjects_dir = get_subjects_dir(raise_error=True)
    if src is None:
        fname_src_fs = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                    'fsaverage-ico-5-src.fif')
        src = read_source_spaces(fname_src_fs)

    picks = np.arange(len(ea.info['ch_names']))

    # Set coord frame
    for idx in range(len(ea.ch_names)):
        ea.info['chs'][idx]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    # Generate source estimate
    kwargs = dict(
        evoked=ea, subject='fsaverage', trans='fsaverage',
        distance=distance, mode=mode, surface=surface,
        subjects_dir=subjects_dir, src=src, project=True)
    stc = stc_near_sensors(picks=picks, **kwargs, verbose=verbose)

    # Produce brain plot
    brain = stc.plot(src=src, subjects_dir=subjects_dir, hemi=hemi,
                     surface=surface, initial_time=0, clim=clim, size=size,
                     colormap=colormap, figure=figure, background=background,
                     colorbar=colorbar, verbose=verbose)
    if view is not None:
        brain.show_view(view)

    return brain
