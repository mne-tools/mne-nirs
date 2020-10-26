# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import nilearn

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.simulation import simulate_nirs_raw


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
    _, ni_est = nilearn.glm.first_level.run_glm(
        raw.get_data(0).T, design_matrix.values)
    assert type(ni_est) == type(glm_estimates)
