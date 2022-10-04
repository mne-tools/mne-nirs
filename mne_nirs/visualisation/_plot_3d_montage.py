# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import contextlib
import inspect

import numpy as np

from mne import pick_info, pick_types
from mne.channels import make_standard_montage
from mne.channels.montage import transform_to_head
from mne.io import Info
from mne.transforms import _get_trans, apply_trans
from mne.utils import _validate_type, _check_option, verbose, logger
from mne.viz import Brain


@verbose
def plot_3d_montage(info, view_map, *, src_det_names='auto',
                    ch_names='numbered', subject='fsaverage',
                    trans='fsaverage', surface='pial',
                    subjects_dir=None, verbose=None):
    """
    Plot a 3D sensor montage.

    Parameters
    ----------
    info : instance of Info
        Measurement info.
    view_map : dict
        Dict of view (key) to channel-pair-numbers (value) to use when
        plotting. Note that, because these get plotted as 1-based channel
        *numbers*, the values should be 1-based rather than 0-based.
        The keys are of the form:

        ``'{side}-{view}'``
            For views like ``'left-lat'`` or ``'right-frontal'`` where the side
            matters.
        ``'{view}'``
            For views like ``'caudal'`` that are along the midline.

        See :meth:`mne.viz.Brain.show_view` for ``view`` options, and the
        Examples section below for usage examples.
    src_det_names : None | dict | str
        Source and detector names to use. "auto" (default) will see if the
        channel locations correspond to standard 10-20 locations and will
        use those if they do (otherwise will act like None). None will use
        S1, S2, ..., D1, D2, ..., etc. Can also be an explicit dict mapping,
        for example::

            src_det_names=dict(S1='Fz', D1='FCz', ...)
    ch_names : str | dict | None
        If ``'numbered'`` (default), use ``['1', '2', ...]`` for the channel
        names, or ``None`` to use ``['S1_D2', 'S2_D1', ...]``. Can also be a
        dict to provide a mapping from the ``'S1_D2'``-style names (keys) to
        other names, e.g., ``defaultdict(lambda: '')`` will prevent showing
        the names altogether.

        .. versionadded:: 0.3
    subject : str
        The subject.
    trans : str | Transform
        The subjects head<->MRI transform.
    surface : str
        The FreeSurfer surface name (e.g., 'pial', 'white').
    subjects_dir : str
        The subjects directory.
    %(verbose)s

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figimage.

    Examples
    --------
    For a Hitachi system with two sets of 12 source-detector arrangements,
    one on each side of the head, showing 1-12 on the left and 13-24 on the
    right can be accomplished using the following ``view_map``::

        >>> view_map = {
        ...     'left-lat': np.arange(1, 13),
        ...     'right-lat': np.arange(13, 25),
        ... }

    NIRx typically involves more complicated arrangements. See
    :ref:`the 3D tutorial <tut-fnirs-vis-brain-plot-3d-montage>` for
    an advanced example that incorporates the ``'caudal'`` view as well.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    _validate_type(info, Info, 'info')
    _validate_type(view_map, dict, 'views')
    _validate_type(src_det_names, (None, dict, str), 'src_det_names')
    _validate_type(ch_names, (dict, str, None), 'ch_names')
    info = pick_info(info, pick_types(info, fnirs=True, exclude=())[::2])
    if isinstance(ch_names, str):
        _check_option('ch_names', ch_names, ('numbered',), extra='when str')
        ch_names = {
            name.split()[0]: str(ni)
            for ni, name in enumerate(info['ch_names'], 1)}
    info['bads'] = []
    if isinstance(src_det_names, str):
        _check_option('src_det_names', src_det_names, ('auto',),
                      extra='when str')
        # Decide if we can map to 10-20 locations
        names, pos = zip(
            *transform_to_head(make_standard_montage('standard_1020'))
            .get_positions()['ch_pos'].items())
        pos = np.array(pos, float)
        locs = dict()
        bad = False
        for ch in info['chs']:
            name = ch['ch_name']
            s_name, d_name = name.split()[0].split('_')
            for name, loc in [(s_name, ch['loc'][3:6]),
                              (d_name, ch['loc'][6:9])]:
                if name in locs:
                    continue
                # see if it's close enough
                idx = np.where(cdist(loc[np.newaxis], pos)[0] < 1e-3)[0]
                if len(idx) < 1:
                    bad = True
                    break
                # Some are duplicated (e.g., T7+T3) but we can rely on the
                # first one being the canonical one
                locs[name] = names[idx[0]]
            if bad:
                break
        if bad:
            src_det_names = None
            logger.info('Could not automatically map source/detector names to '
                        '10-20 locations.')
        else:
            src_det_names = locs
            logger.info('Source-detector names automatically mapped to 10-20 '
                        'locations')

    head_mri_t = _get_trans(trans, 'head', 'mri')[0]
    del trans
    views = list()
    for key, num in view_map.items():
        _validate_type(key, str, f'view_map key {repr(key)}')
        _validate_type(num, np.ndarray, f'view_map[{repr(key)}]')
        if '-' in key:
            hemi, v = key.split('-', maxsplit=1)
            hemi = dict(left='lh', right='rh')[hemi]
            views.append((hemi, v, num))
        else:
            views.append(('lh', key, num))
    del view_map
    size = (400 * len(views), 400)
    brain = Brain(
        subject, 'both', surface, views=['lat'] * len(views),
        size=size, background='w', units='m',
        view_layout='horizontal', subjects_dir=subjects_dir)
    with _safe_brain_close(brain):
        brain.add_head(dense=False, alpha=0.1)
        brain.add_sensors(
            info, trans=head_mri_t,
            fnirs=['channels', 'pairs', 'sources', 'detectors'])
        add_text_kwargs = dict()
        if 'render' in inspect.signature(brain.plotter.add_text).parameters:
            add_text_kwargs['render'] = False
        for col, view in enumerate(views):
            plotted = set()
            brain.show_view(
                view[1], hemi=view[0], focalpoint=(0, -0.02, 0.02),
                distance=0.4, row=0, col=col)
            brain.plotter.subplot(0, col)
            vp = brain.plotter.renderer
            for ci in view[2]:  # figure out what we need to plot
                this_ch = info['chs'][ci - 1]
                ch_name = this_ch['ch_name'].split()[0]
                s_name, d_name = ch_name.split('_')
                needed = [
                    (ch_names, 'ch_names', ch_name,
                     this_ch['loc'][:3], 12, 'Centered'),
                    (src_det_names, 'src_det_names', s_name,
                     this_ch['loc'][3:6], 8, 'Bottom'),
                    (src_det_names, 'src_det_names', d_name,
                     this_ch['loc'][6:9], 8, 'Bottom'),
                ]
                for lookup, lname, name, ch_pos, font_size, va in needed:
                    if name in plotted:
                        continue
                    plotted.add(name)
                    orig_name = name
                    if lookup is not None:
                        name = lookup[name]
                    _validate_type(name, str, f'{lname}[{repr(orig_name)}]')
                    ch_pos = apply_trans(head_mri_t, ch_pos)
                    vp.SetWorldPoint(np.r_[ch_pos, 1.])
                    vp.WorldToDisplay()
                    ch_pos = (np.array(vp.GetDisplayPoint()[:2]) -
                              np.array(vp.GetOrigin()))
                    actor = brain.plotter.add_text(
                        name, ch_pos, font_size=font_size, color=(0., 0., 0.),
                        **add_text_kwargs)
                    prop = actor.GetTextProperty()
                    getattr(prop, f'SetVerticalJustificationTo{va}')()
                    prop.SetJustificationToCentered()
                    actor.SetTextProperty(prop)
                    prop.SetBold(True)
        img = brain.screenshot()
    return plt.figimage(img, resize=True).figure


@contextlib.contextmanager
def _safe_brain_close(brain):
    try:
        yield
    finally:
        brain.close()
