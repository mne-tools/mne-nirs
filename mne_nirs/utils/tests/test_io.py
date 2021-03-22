# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import os
import pytest
import mne
import mne_nirs
import numpy as np

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.utils._io import glm_to_tidy, _tidy_long_to_wide
from mne_nirs.statistics import run_GLM


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_io():
    num_chans = 6
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
    raw_intensity.resample(0.2)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_haemo = mne_nirs.channels.get_long_channels(raw_haemo)
    raw_haemo.pick(picks=range(num_chans))
    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   hrf_model='spm',
                                                   stim_dur=5.0,
                                                   drift_order=3,
                                                   drift_model='polynomial')
    glm_est = run_GLM(raw_haemo, design_matrix)
    df = glm_to_tidy(raw_haemo, glm_est, design_matrix)
    assert df.shape == (48, 12)
    assert set(df.columns) == {'ch_name', 'Condition', 'df', 'mse', 'p_value',
                               't', 'theta', 'Source', 'Detector', 'Chroma',
                               'Significant', 'se'}
    num_conds = 8  # triggers (1, 2, 3, 15) + 3 drifts + constant
    assert df.shape[0] == num_chans * num_conds
    assert len(df["se"]) == 48
    assert sum(df["se"]) > 0  # Check isn't nan
    assert len(df["df"]) == 48
    assert sum(df["df"]) > 0  # Check isn't nan
    assert len(df["p_value"]) == 48
    assert sum(df["p_value"]) > 0  # Check isn't nan
    assert len(df["theta"]) == 48
    assert sum(df["theta"]) > 0  # Check isn't nan
    assert len(df["t"]) == 48
    assert sum(df["t"]) > -99999  # Check isn't nan

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['2.0'] - basic_conts['3.0']

    contrast = mne_nirs.statistics.compute_contrast(glm_est, contrast_LvR)
    df = glm_to_tidy(raw_haemo, contrast, design_matrix)
    assert df.shape == (6, 10)
    assert set(df.columns) == {'ch_name', 'ContrastType', 'z_score', 'stat',
                               'p_value', 'effect', 'Source', 'Detector',
                               'Chroma', 'Significant'}

    contrast = mne_nirs.statistics.compute_contrast(glm_est, contrast_LvR,
                                                    contrast_type='F')
    df = glm_to_tidy(raw_haemo, contrast, design_matrix, wide=False)
    df = _tidy_long_to_wide(df)
    assert df.shape == (6, 10)
    assert set(df.columns) == {'ch_name', 'ContrastType', 'z_score', 'stat',
                               'p_value', 'effect', 'Source', 'Detector',
                               'Chroma', 'Significant'}

    with pytest.raises(TypeError, match="Unknown statistic type"):
        glm_to_tidy(raw_haemo, [1, 2, 3], design_matrix, wide=False)
