# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import pytest
import mne
import mne_nirs
import numpy as np
from mne.utils import (requires_pysurfer, traits_test, requires_mayavi)
from mne_nirs.experimental_design.tests.test_experimental_design import \
    _load_dataset
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.visualisation import plot_glm_topo


@requires_pysurfer
@traits_test
def test_plot_nirs_source_detector_pyvista():
    mne.viz.set_3d_backend('pyvista')
    data_path = mne.datasets.testing.data_path() + '/NIRx/nirscout'
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    raw = mne.io.read_raw_nirx(data_path + '/nirx_15_2_recording_w_short')

    mne_nirs.visualisation.plot_nirs_source_detector(
        np.random.randn(len(raw.ch_names)),
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['brain'],
        fnirs=False,
        subjects_dir=subjects_dir,
        verbose=True)

    mne_nirs.visualisation.plot_nirs_source_detector(
        np.abs(np.random.randn(len(raw.ch_names))) + 5,
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['brain'],
        fnirs=False,
        subjects_dir=subjects_dir,
        verbose=True)


@requires_mayavi
@traits_test
def test_plot_nirs_source_detector_mayavi():
    mne.viz.set_3d_backend('mayavi')
    data_path = mne.datasets.testing.data_path() + '/NIRx/nirscout'
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    raw = mne.io.read_raw_nirx(data_path + '/nirx_15_2_recording_w_short')

    mne_nirs.visualisation.plot_nirs_source_detector(
        np.random.randn(len(raw.ch_names)),
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['brain'],
        fnirs=False,
        cmap='inferno',
        subjects_dir=subjects_dir,
        verbose=True)


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_run_plot_GLM_topo():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    glm_estimates = run_GLM(raw_haemo, design_matrix)
    fig = plot_glm_topo(raw_haemo, glm_estimates, design_matrix)
    # 5 conditions (A,B,C,Drift,Constant) * two chroma + 2xcolorbar
    assert len(fig.axes) == 12

    # Two conditions * two chroma + 2 x colorbar
    fig = plot_glm_topo(raw_haemo, glm_estimates, design_matrix,
                        requested_conditions=['A', 'B'])
    assert len(fig.axes) == 6

    # Two conditions * one chroma + 1 x colorbar
    with pytest.warns(RuntimeWarning, match='Reducing GLM results'):
        fig = plot_glm_topo(raw_haemo.copy().pick(picks="hbo"),
                            glm_estimates, design_matrix,
                            requested_conditions=['A', 'B'])
    assert len(fig.axes) == 3

    # One conditions * two chroma + 2 x colorbar
    fig = plot_glm_topo(raw_haemo, glm_estimates, design_matrix,
                        requested_conditions=['A'])
    assert len(fig.axes) == 4

    # One conditions * one chroma + 1 x colorbar
    with pytest.warns(RuntimeWarning, match='Reducing GLM results'):
        fig = plot_glm_topo(raw_haemo.copy().pick(picks="hbo"),
                            glm_estimates,
                            design_matrix, requested_conditions=['A'])
    assert len(fig.axes) == 2

    # One conditions * one chroma + 0 x colorbar
    with pytest.warns(RuntimeWarning, match='Reducing GLM results'):
        fig = plot_glm_topo(raw_haemo.copy().pick(picks="hbo"),
                            glm_estimates, design_matrix,
                            colorbar=False, requested_conditions=['A'])
    assert len(fig.axes) == 1

    # Ensure warning thrown if glm estimates is missing channels from raw
    glm_estimates_subset = {a: glm_estimates[a]
                            for a in raw_haemo.ch_names[0:3]}
    with pytest.raises(RuntimeError, match="does not match regression"):
        plot_glm_topo(raw_haemo, glm_estimates_subset, design_matrix)


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_run_plot_GLM_contrast_topo():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    glm_est = run_GLM(raw_haemo, design_matrix)
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['A'] - basic_conts['B']
    contrast = mne_nirs.statistics.compute_contrast(glm_est, contrast_LvR)
    fig = mne_nirs.visualisation.plot_glm_contrast_topo(raw_haemo, contrast)
    assert len(fig.axes) == 3


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_run_plot_GLM_contrast_topo_single_chroma():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast

    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_haemo = raw_haemo.pick(picks='hbo')
    glm_est = run_GLM(raw_haemo, design_matrix)
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['A'] - basic_conts['B']
    contrast = mne_nirs.statistics.compute_contrast(glm_est, contrast_LvR)
    fig = mne_nirs.visualisation.plot_glm_contrast_topo(raw_haemo, contrast)
    assert len(fig.axes) == 2


def test_fig_from_axes():
    from mne_nirs.visualisation._plot_GLM_topo import _get_fig_from_axes
    with pytest.raises(RuntimeError, match="Unable to extract figure"):
        _get_fig_from_axes([1, 2, 3])
