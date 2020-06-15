# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import numpy as np
import mne
import nilearn

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.simulation import simulate_nirs_raw
from mne_nirs.channels import get_short_channels, get_long_channels
from mne_nirs.utils._io import GLM_to_tidy_long, _tidy_long_to_wide


def test_run_GLM():
    raw = simulate_nirs_raw(sig_dur=200, stim_dur=5.)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    glm_estimates = run_GLM(raw, design_matrix)

    assert len(glm_estimates) == len(raw.ch_names)

    # Check the estimate is correct within 10% error
    assert abs(glm_estimates["Simulated"].theta[0] - 1.e-6) < 0.1e-6

    # ensure we return the same type as nilearn to encourage compatibility
    _, ni_est = nilearn.stats.first_level_model.run_glm(
        raw.get_data(0).T, design_matrix.values)
    assert type(ni_est) == type(glm_estimates)


def test_GLM_system_test():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
    raw_intensity.resample(1.0)
    new_des = [des for des in raw_intensity.annotations.description]
    new_des = ['Control' if x == "1.0" else x for x in new_des]
    new_des = ['Tapping/Left' if x == "2.0" else x for x in new_des]
    new_des = ['Tapping/Right' if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(raw_intensity.annotations.onset,
                            raw_intensity.annotations.duration, new_des)
    raw_intensity.set_annotations(annot)
    raw_intensity.annotations.crop(35, 2967)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    short_chs = get_short_channels(raw_haemo)
    raw_haemo = get_long_channels(raw_haemo)
    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   hrf_model='spm',
                                                   stim_dur=5.0,
                                                   drift_order=3,
                                                   drift_model='polynomial')
    design_matrix["ShortHbO"] = np.mean(short_chs.copy().pick(
        picks="hbo").get_data(), axis=0)
    design_matrix["ShortHbR"] = np.mean(short_chs.copy().pick(
        picks="hbr").get_data(), axis=0)
    glm_est = run_GLM(raw_haemo, design_matrix)
    df = GLM_to_tidy_long(raw_haemo, glm_est, design_matrix)
    df = _tidy_long_to_wide(df)
    a = (df
         .query('condition in ["Control"]')
         .groupby(['condition', 'Chroma'])
         .agg(['mean'])
         )
    # Make sure false positive rate is less than 5%
    assert a["Significant"].values[0] < 0.05
    assert a["Significant"].values[1] < 0.05
    a = (df
         .query('condition in ["Tapping/Left", "Tapping/Right"]')
         .groupby(['condition', 'Chroma'])
         .agg(['mean'])
         )
    # Fairly arbitrary cutoff here, but its more than 5%
    assert a["Significant"].values[0] > 0.7
    assert a["Significant"].values[1] > 0.7
    assert a["Significant"].values[2] > 0.7
    assert a["Significant"].values[3] > 0.7
