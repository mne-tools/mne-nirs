# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from collections import defaultdict

import pytest
import numpy as np
import mne
import mne_nirs

from mne.datasets import testing
from mne.utils import catch_logging, check_version

from mne_nirs.experimental_design.tests.test_experimental_design import \
    _load_dataset
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.visualisation import plot_glm_surface_projection
from mne_nirs.utils import glm_to_tidy
from mne_nirs.statistics.tests.test_glm_type import _get_glm_result


testing_path = testing.data_path(download=False)
raw_path = str(testing_path) + '/NIRx/nirscout/nirx_15_2_recording_w_short'
subjects_dir = str(testing_path) + '/subjects'

requires_mne_1 = pytest.mark.skipif(not check_version('mne', '1.0'),
                                    reason='Needs MNE-Python 1.0')


def test_plot_nirs_source_detector_pyvista(requires_pyvista):
    raw = mne.io.read_raw_nirx(raw_path)

    mne_nirs.visualisation.plot_nirs_source_detector(
        np.random.randn(len(raw.ch_names)),
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['white'],
        fnirs=False,
        subjects_dir=subjects_dir,
        verbose=True)

    mne_nirs.visualisation.plot_nirs_source_detector(
        np.abs(np.random.randn(len(raw.ch_names))) + 5,
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['white'],
        fnirs=False,
        subjects_dir=subjects_dir,
        verbose=True)


def test_run_plot_GLM_topo():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    glm_estimates = run_glm(raw_haemo, design_matrix)
    fig = glm_estimates.plot_topo()
    # 5 conditions (A,B,C,Drift,Constant) * two chroma + 2xcolorbar
    assert len(fig.axes) == 12

    # Two conditions * two chroma + 2 x colorbar
    fig = glm_estimates.plot_topo(conditions=['A', 'B'])
    assert len(fig.axes) == 6

    # One conditions * two chroma + 2 x colorbar
    fig = glm_estimates.plot_topo(conditions=['A'])
    assert len(fig.axes) == 4


def test_run_plot_GLM_contrast_topo():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    glm_est = run_glm(raw_haemo, design_matrix)
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['A'] - basic_conts['B']
    contrast = glm_est.compute_contrast(contrast_LvR)
    fig = contrast.plot_topo()
    assert len(fig.axes) == 3


def test_run_plot_GLM_contrast_topo_single_chroma():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    raw_haemo = raw_haemo.pick(picks='hbo')
    glm_est = run_glm(raw_haemo, design_matrix)
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['A'] - basic_conts['B']
    contrast = glm_est.compute_contrast(contrast_LvR)
    fig = contrast.plot_topo()
    assert len(fig.axes) == 2


def test_fig_from_axes():
    from mne_nirs.visualisation._plot_GLM_topo import _get_fig_from_axes
    with pytest.raises(RuntimeError, match="Unable to extract figure"):
        _get_fig_from_axes([1, 2, 3])


# surface arg
@requires_mne_1
def test_run_plot_GLM_projection(requires_pyvista):
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    glm_estimates = run_glm(raw_haemo, design_matrix)
    df = glm_to_tidy(raw_haemo, glm_estimates.data, design_matrix)
    df = df.query("Chroma in 'hbo'")
    df = df.query("Condition in 'A'")

    brain = plot_glm_surface_projection(raw_haemo.copy().pick("hbo"),
                                        df, clim='auto', view='dorsal',
                                        colorbar=True, size=(800, 700),
                                        value="theta", surface='white',
                                        subjects_dir=subjects_dir)
    assert type(brain) == mne.viz._brain.Brain


@requires_mne_1
@pytest.mark.parametrize('fname_raw, to_1020, ch_names', [
    (raw_path, False, None),
    (raw_path, True, 'numbered'),
    (raw_path, True, defaultdict(lambda: '')),
])
def test_plot_3d_montage(requires_pyvista, fname_raw, to_1020, ch_names):
    import pyvista
    pyvista.close_all()
    assert len(pyvista.plotting._ALL_PLOTTERS) == 0
    raw = mne.io.read_raw_nirx(fname_raw)
    if to_1020:
        need = set(sum(
            (ch_name.split()[0].split('_') for ch_name in raw.ch_names),
            list()))
        mon = mne.channels.make_standard_montage('standard_1020')
        mon.rename_channels({h: n for h, n in zip(mon.ch_names, need)})
        raw.set_montage(mon)
    n_labels = len(raw.ch_names) // 2
    view_map = {'left-lat': np.arange(1, n_labels // 2),
                'caudal': np.arange(n_labels // 2, n_labels + 1)}
    # We use "sample" here even though it's wrong so that we can have a head
    # surface
    with catch_logging() as log:
        mne_nirs.viz.plot_3d_montage(
            raw.info, view_map, subject='sample', surface='white',
            subjects_dir=subjects_dir, ch_names=ch_names, verbose=True)
    assert len(pyvista.plotting._ALL_PLOTTERS) == 0
    log = log.getvalue().lower()
    if to_1020:
        assert 'automatically mapped' in log
    else:
        assert 'could not' in log


# surface arg
@pytest.mark.skipif(not check_version('mne', '1.0'),
                    reason='Needs MNE-Python 1.0')
def test_glm_surface_projection(requires_pyvista):

    res = _get_glm_result(tmax=2974, tmin=0)
    res.surface_projection(condition="e3p0", view="dorsal", surface="white",
                           subjects_dir=subjects_dir)
    with pytest.raises(KeyError, match='not found in conditions'):
        res.surface_projection(condition='foo')
