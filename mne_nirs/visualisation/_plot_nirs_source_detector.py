# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import numpy as np

from mne.viz import plot_alignment
from mne import verbose


@verbose
def plot_nirs_source_detector(data, info=None, radius=0.001,
                              trans=None, subject=None,
                              subjects_dir=None,
                              surfaces='head', coord_frame='head',
                              meg=None, eeg='original', fwd=None,
                              dig=False, ecog=True, src=None,
                              mri_fiducials=False,
                              bem=None, seeg=True, fnirs=False,
                              show_axes=False,
                              fig=None, cmap=None,
                              interaction='trackball', verbose=None):
    """
    3D visualisation of fNIRS response magnitude.

    This function plots the response amplitude for each channel.
    Each channel is represented by a line between the source and detector,
    the color of the line reflects the response magnitude.

    Parameters
    ----------
    data : array
        Array of values to be plotted between source and detectors.
        One value should be specified per channel in the same order
        as `info.chs`.
    info : dict | None
        The measurement info.
        If None (default), no sensor information will be shown.
    radius : numbers
        Tube radius for connecting links.
    %(trans)s
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. Can be omitted if ``src`` is provided.
    %(subjects_dir)s
    surfaces : str | list
        Surfaces to plot. Supported values:

        * scalp: one of 'head', 'outer_skin' (alias for 'head'),
          'head-dense', or 'seghead' (alias for 'head-dense')
        * skull: 'outer_skull', 'inner_skull', 'brain' (alias for
          'inner_skull')
        * brain: one of 'pial', 'white', 'inflated', or 'brain'
          (alias for 'pial').

        Defaults to 'head'.

        .. note:: For single layer BEMs it is recommended to use 'brain'.
    coord_frame : str
        Coordinate frame to use, 'head', 'meg', or 'mri'.
    meg : str | list | bool | None
        Can be "helmet", "sensors" or "ref" to show the MEG helmet, sensors or
        reference sensors respectively, or a combination like
        ``('helmet', 'sensors')`` (same as None, default). True translates to
        ``('helmet', 'sensors', 'ref')``.
    eeg : bool | str | list
        String options are:

        - "original" (default; equivalent to ``True``)
            Shows EEG sensors using their digitized locations (after
            transformation to the chosen ``coord_frame``)
        - "projected"
            The EEG locations projected onto the scalp, as is done in forward
            modeling

        Can also be a list of these options, or an empty list (``[]``,
        equivalent of ``False``).
    fwd : instance of Forward
        The forward solution. If present, the orientations of the dipoles
        present in the forward solution are displayed.
    dig : bool | 'fiducials'
        If True, plot the digitization points; 'fiducials' to plot fiducial
        points only.
    ecog : bool
        If True (default), show ECoG sensors.
    src : instance of SourceSpaces | None
        If not None, also plot the source space points.
    mri_fiducials : bool | str
        Plot MRI fiducials (default False). If ``True``, look for a file with
        the canonical name (``bem/{subject}-fiducials.fif``). If ``str`` it
        should provide the full path to the fiducials file.
    bem : list of dict | instance of ConductorModel | None
        Can be either the BEM surfaces (list of dict), a BEM solution or a
        sphere model. If None, we first try loading
        `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and then look for
        `'$SUBJECT*$SOURCE.fif'` in the same directory. For `'outer_skin'`,
        the subjects bem and bem/flash folders are searched. Defaults to None.
    seeg : bool
        If True (default), show sEEG electrodes.
    fnirs : bool
        If True (default), show fNIRS electrodes.
    show_axes : bool
        If True (default False), coordinate frame axis indicators will be
        shown:

        * head in pink.
        * MRI in gray (if ``trans is not None``).
        * MEG in blue (if MEG sensors are present).

        .. versionadded:: 0.16
    fig : mayavi.mlab.Figure | None
        Mayavi Scene in which to plot the alignment.
        If ``None``, creates a new 600x600 pixel figure with black background.

        .. versionadded:: 0.16
    cmap : str
        Colormap to be used.
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.

        .. versionadded:: 0.16
    %(verbose)s

    Returns
    -------
    fig : Figure
        The 3D figure.

    Notes
    -----
    For more information see :func:`mne.viz.plot_alignment`.

    .. versionadded:: 0.15
    """
    # Determine range of values for creating colormap
    vmin = np.min(data)
    vmax = np.max(data)

    # If no colormap is specified choose depending on range of data
    if cmap is None:
        if (vmin >= 0) & (vmax >= 0):
            # For positive only data use magma
            cmap = 'Oranges'
        else:
            # Otherwise use blue to red and ensure zero sits at white
            vmin = -1. * np.max(np.abs(data))
            vmax = np.max(np.abs(data))
            cmap = 'RdBu_r'

    if isinstance(radius, (int, float)):
        radius = np.ones(len(info['chs'])) * radius

    # Plot requested alignment
    fig = plot_alignment(
        info=info, trans=trans, subject=subject,
        subjects_dir=subjects_dir,
        surfaces=surfaces, coord_frame=coord_frame,
        meg=meg, eeg=eeg, fwd=fwd,
        dig=dig, ecog=ecog, src=src,
        mri_fiducials=mri_fiducials,
        bem=bem, seeg=seeg, fnirs=fnirs,
        show_axes=show_axes,
        fig=fig,
        interaction=interaction, verbose=verbose)

    from mne.viz.backends.renderer import _get_renderer
    renderer = _get_renderer(fig)

    # Overlay channels between source and detectors
    for idx, ch in enumerate(info['chs']):
        locs = ch['loc']

        renderer.tube(origin=[np.array([locs[3], locs[4], locs[5]])],
                      destination=[np.array([locs[6], locs[7], locs[8]])],
                      scalars=np.array([[1.0, 1.0]]) * data[idx],
                      radius=radius[idx], colormap=cmap,
                      vmin=vmin, vmax=vmax)

    t = renderer.tube(origin=[np.array([0, 0, 0])],
                      destination=[np.array([0, 0, 0.001])],
                      scalars=np.array([[vmin, vmax]]),
                      radius=0.0001, colormap=cmap,
                      vmin=vmin, vmax=vmax)
    renderer.scalarbar(t)

    return fig
