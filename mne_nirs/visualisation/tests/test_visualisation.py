# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne
import mne_nirs
import numpy as np
from mne.utils import (requires_pysurfer, traits_test)
from mne_nirs.experimental_design.tests.test_experimental_design import \
    _load_dataset
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.visualisation import plot_glm_topo

mne.viz.set_3d_backend('pyvista')


@requires_pysurfer
@traits_test
def test_plot_nirs_source_detector():
    data_path = mne.datasets.testing.data_path()
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    raw = mne.io.read_raw_nirx(data_path + '/NIRx/nirx_15_2_recording_w_short')

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
        100.0 + np.random.randn(len(raw.ch_names)),
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['brain'],
        fnirs=False,
        subjects_dir=subjects_dir,
        verbose=True)

    mne_nirs.visualisation.plot_nirs_source_detector(
        100.0 + np.random.randn(len(raw.ch_names)),
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['brain'],
        fnirs=False, cmap='viridis',
        subjects_dir=subjects_dir,
        verbose=True)


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

    fig = plot_glm_topo(raw_haemo, glm_estimates, design_matrix,
                        requested_conditions=['A', 'B'])
    # Two conditions * two chroma + 2xcolorbar
    assert len(fig.axes) == 6


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
