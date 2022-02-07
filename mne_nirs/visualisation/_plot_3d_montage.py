# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import contextlib
import numpy as np

from mne import pick_info, pick_types
from mne.channels import make_standard_montage
from mne.channels.montage import transform_to_head
from mne.fixes import _get_args
from mne.io import Info
from mne.transforms import _get_trans, apply_trans
from mne.utils import _validate_type, _check_option, verbose, logger
from mne.viz import Brain


@verbose
def plot_3d_montage(info, view_map, *, src_det_names='auto',
                    subject='fsaverage', trans='fsaverage',
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

        See :meth:`mne.viz.Brain.show_view` for ``view`` options.
    src_det_names : None | dict | str
        Source and detector names to use. "auto" (default) will see if the
        channel locations correspond to standard 10-20 locations and will
        use those if they do (otherwise will act like None). None will use
        S1, S2, ..., D1, D2, ..., etc. Can also be an explicit dict mapping.
    subject : str
        The subject.
    trans : str | Transform
        The subjects head<->MRI transform.
    subjects_dir : str
        The subjects directory.
    %(verbose)s

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figimage.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    _validate_type(info, Info, 'info')
    _validate_type(view_map, dict, 'views')
    _validate_type(src_det_names, (None, dict, str), 'src_det_names')
    info = pick_info(info, pick_types(info, fnirs=True, exclude=())[::2])
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

    trans = _get_trans('fsaverage', 'head', 'mri')[0]
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
        subject, 'both', 'pial', views=['lat'] * len(views),
        size=size, background='w', units='m',
        view_layout='horizontal')
    with _safe_brain_close(brain):
        brain.add_head(dense=False, alpha=0.1)
        brain.add_sensors(
            info, trans='fsaverage',
            fnirs=['channels', 'pairs', 'sources', 'detectors'])
        add_text_kwargs = dict()
        if 'render' in _get_args(brain.plotter.add_text):
            add_text_kwargs['render'] = False
        for col, view in enumerate(views):
            plotted = set()
            brain.show_view(
                view[1], hemi=view[0], focalpoint=(0, -0.02, 0.02),
                distance=0.4, row=0, col=col)
            brain.plotter.subplot(0, col)
            vp = brain.plotter.renderer
            for ci in view[2]:  # figure out what we need to plot
                ch_name = str(ci)
                this_ch = info['chs'][ci - 1]
                name = this_ch['ch_name']
                s_name, d_name = name.split()[0].split('_')
                if src_det_names is not None:
                    s_name = src_det_names[s_name]
                    d_name = src_det_names[d_name]
                needed = [
                    (ch_name, this_ch['loc'][:3], 12, 'Centered'),
                    (s_name, this_ch['loc'][3:6], 8, 'Bottom'),
                    (d_name, this_ch['loc'][6:9], 8, 'Bottom'),
                ]
                for name, ch_pos, font_size, va in needed:
                    if name in plotted:
                        continue
                    plotted.add(name)
                    # name = rev_dict[use_]  # XXX ADD THIS
                    ch_pos = apply_trans(trans, ch_pos)
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
        try:
            brain.close()
        except Exception:
            pass
