# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import pandas
import pytest
import numpy as np
import matplotlib
from matplotlib.pyplot import Axes

import mne
import nilearn

from mne_nirs.statistics import RegressionResults, read_glm
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm


def _get_minimal_haemo_data(tmin=0, tmax=60):
    raw = mne.io.read_raw_nirx(os.path.join(
        mne.datasets.fnirs_motor.data_path(), 'Participant-1'), preload=False)
    raw.crop(tmax=tmax, tmin=tmin)
    raw = mne.preprocessing.nirs.optical_density(raw)
    raw = mne.preprocessing.nirs.beer_lambert_law(raw)
    raw.resample(0.3)
    return raw


def _get_glm_result(tmax=60, tmin=0, noise_model='ar1'):
    raw = _get_minimal_haemo_data(tmin=tmin, tmax=tmax)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    return run_glm(raw, design_matrix, noise_model=noise_model)


def _get_glm_contrast_result(tmin=60, tmax=400):
    raw = _get_minimal_haemo_data(tmin=tmin, tmax=tmax)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.,
                                                   drift_order=1,
                                                   drift_model='polynomial')

    glm_est = run_glm(raw, design_matrix)

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['1.0'] - basic_conts['2.0']

    return glm_est.compute_contrast(contrast_LvR)


def test_create_results_glm():

    # Create a relevant info structure
    raw = _get_minimal_haemo_data()

    # Create a minimal structure
    res = _get_glm_result()

    # Get arguments for type so we can test errors below
    info = raw.info
    minimal_structure = res._data

    # Test construction
    with pytest.raises(TypeError, match='must be a dictionary'):
        _ = RegressionResults(info, np.zeros((5, 2)), 1)
    with pytest.raises(TypeError, match='must be a dictionary'):
        _ = RegressionResults(info, 3.2, 1)
    with pytest.raises(TypeError, match='must be a dictionary'):
        _ = RegressionResults(info, [], 1)
    with pytest.raises(TypeError, match='must be a dictionary'):
        _ = RegressionResults(info, "hbo", 1)
    with pytest.raises(TypeError, match='keys must match'):
        _ = RegressionResults(info, _take(4, minimal_structure), 1)

    onewrongname = _take(55, minimal_structure)
    onewrongname["test"] = onewrongname["S1_D1 hbo"]
    with pytest.raises(TypeError, match='must match ch_names'):
        _ = RegressionResults(info, onewrongname, 1)

    # Test properties
    assert len(res) == len(raw.ch_names)


def test_results_glm_properties():

    n_channels = 56

    res = _get_glm_result()

    # Test ContainsMixin
    assert 'hbo' in res
    assert 'hbr' in res
    assert 'meg' not in res

    # Test copy
    assert len(res) == len(res.copy())

    # Test picks
    assert len(res.copy().pick(picks=range(4))) == 4
    assert len(res.copy().pick(picks="S1_D1 hbr")) == 1
    assert len(res.copy().pick(picks=["S1_D1 hbr"])) == 1
    assert len(res.copy().pick(picks=["S1_D1 hbr", "S1_D1 hbo"])) == 2
    assert len(res.copy().pick(picks=["S1_D1 hbr", "S1_D1 XXX"])) == 1
    assert len(res.copy().pick(picks=["S1_D1 hbr", "S1_D1 hbr"])) == 1
    assert len(res.copy().pick(picks="fnirs")) == n_channels
    assert len(res.copy().pick(picks="hbo")) == n_channels / 2
    assert len(res.copy().pick(picks="hbr")) == n_channels / 2

    # Test results
    assert len(res.theta()) == n_channels
    assert len(res.copy().pick(picks=range(4)).theta()) == 4
    assert len(res.copy().pick(picks=3).theta()) == 1
    assert res.copy().pick(picks=3).theta()[0].shape == (3, 1)

    # Test models
    assert len(res.model()) == n_channels
    assert len(res.copy().pick(picks=range(8)).model()) == 8
    assert type(res.model()[0]) is nilearn.glm.regression.ARModel

    assert isinstance(res.to_dataframe(), pandas.DataFrame)


def test_glm_scatter():

    assert isinstance(_get_glm_result().scatter(), Axes)
    assert isinstance(_get_glm_contrast_result().scatter(), Axes)
    _get_glm_result(tmax=2974, tmin=0).surface_projection(condition="3.0",
                                                          view="dorsal")


def test_results_glm_export_dataframe():

    n_channels = 56
    res = _get_glm_result(tmax=400)
    df = res.to_dataframe()

    assert df.shape == (6 * n_channels, 12)


def test_create_results_glm_contrast():

    # Create a minimal structure
    res = _get_glm_contrast_result()
    assert isinstance(res._data, nilearn.glm.contrasts.Contrast)
    assert isinstance(res.info, mne.Info)

    # Test copy
    assert len(res) == len(res.copy())

    assert isinstance(res.plot_topo(), matplotlib.figure.Figure)

    n_channels = 56
    assert isinstance(res.to_dataframe(), pandas.DataFrame)
    df = res.to_dataframe()
    assert df.shape == (n_channels, 10)


def test_results_glm_io():

    res = _get_glm_result(tmax=400)
    res.save("test-regression-glm.h5", overwrite=True)
    loaded_res = read_glm("test-regression-glm.h5")
    assert loaded_res.to_dataframe().equals(res.to_dataframe())
    assert res == loaded_res

    res = _get_glm_result(tmax=400, noise_model='ols')
    res.save("test-regression-ols_glm.h5", overwrite=True)
    loaded_res = read_glm("test-regression-ols_glm.h5")
    assert loaded_res.to_dataframe().equals(res.to_dataframe())
    assert res == loaded_res

    res = _get_glm_contrast_result()
    res.save("test-contrast-glm.h5", overwrite=True)
    loaded_res = read_glm("test-contrast-glm.h5")
    assert loaded_res.to_dataframe().equals(res.to_dataframe())
    assert res == loaded_res

    with pytest.raises(IOError, match='must end with glm.h5'):
        res.save("test-contrast-glX.h5", overwrite=True)


def _take(n, mydict):
    """Return first n items of the iterable as a list"""
    return {k: mydict[k] for k in list(mydict)[:n]}
