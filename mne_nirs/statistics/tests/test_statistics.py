# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.simulation import simulate_nirs_raw


def test_run_GLM():
    raw = simulate_nirs_raw(sig_dur=200, stim_dur=5.)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    labels, glm_estimates = run_GLM(raw, design_matrix)

    assert len(labels) == len(raw.ch_names)

    # the estimates are nested. so cycle through to check correct number
    # are generated
    num = 0
    for est in glm_estimates:
        num += glm_estimates[est].theta.shape[1]
    assert num == len(raw.ch_names)

    # Check the estimate is correct within 10% error
    assert abs(glm_estimates[labels[0]].theta[0] - 1.e-6) < 0.1e-6
