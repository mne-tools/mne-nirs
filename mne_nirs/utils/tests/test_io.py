# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import os
import mne
import mne_nirs

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.utils._io import _GLM_to_tidy_long, _tidy_long_to_wide
from mne_nirs.statistics import run_GLM


def test_io():
    num_chans = 6
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
    raw_intensity.resample(0.2)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_haemo = mne_nirs.utils.get_long_channels(raw_haemo)
    raw_haemo.pick(picks=range(num_chans))
    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   hrf_model='spm',
                                                   stim_dur=5.0,
                                                   drift_order=3,
                                                   drift_model='polynomial')
    labels, glm_est = run_GLM(raw_haemo, design_matrix)
    df = _GLM_to_tidy_long(raw_haemo, labels, glm_est, design_matrix)
    df = _tidy_long_to_wide(df)
    assert df.shape == (48, 11)
    assert set(df.columns) == {'ch_name', 'condition', 'df', 'mse', 'p', 't',
                               'theta', 'Source', 'Detector', 'Chroma',
                               'Significant'}
    num_conds = 8  # triggers (1, 2, 3, 15) + 3 drifts + constant
    assert df.shape[0] == num_chans * num_conds
