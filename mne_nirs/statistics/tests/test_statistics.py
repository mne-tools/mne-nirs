# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import pytest
import numpy as np
import nilearn

from mne import Covariance
from mne.simulation import add_noise

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.simulation import simulate_nirs_raw

iir_filter = [1., -0.58853134, -0.29575669, -0.52246482, 0.38735476, 0.024286]


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
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


def test_run_GLM_order():
    raw = simulate_nirs_raw(sig_dur=200, stim_dur=5., sfreq=3)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.,
                                                   drift_order=1,
                                                   drift_model='polynomial')

    # Default should be first order AR
    glm_estimates = run_GLM(raw, design_matrix)
    assert glm_estimates['Simulated'].model.order == 1

    # Default should be first order AR
    glm_estimates = run_GLM(raw, design_matrix, noise_model='ar2')
    assert glm_estimates['Simulated'].model.order == 2

    glm_estimates = run_GLM(raw, design_matrix, noise_model='ar7')
    assert glm_estimates['Simulated'].model.order == 7

    # Auto should be 4 times sample rate
    cov = Covariance(np.ones(1) * 1e-11, raw.ch_names,
                     raw.info['bads'], raw.info['projs'], nfree=0)
    raw = add_noise(raw, cov, iir_filter=iir_filter)
    glm_estimates = run_GLM(raw, design_matrix, noise_model='auto')
    assert glm_estimates['Simulated'].model.order == 3 * 4

    raw = simulate_nirs_raw(sig_dur=10, stim_dur=5., sfreq=2)
    cov = Covariance(np.ones(1) * 1e-11, raw.ch_names,
                     raw.info['bads'], raw.info['projs'], nfree=0)
    raw = add_noise(raw, cov, iir_filter=iir_filter)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    # Auto should be 4 times sample rate
    glm_estimates = run_GLM(raw, design_matrix, noise_model='auto')
    assert glm_estimates['Simulated'].model.order == 2 * 4
