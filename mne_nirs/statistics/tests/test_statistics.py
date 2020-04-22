# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import mne
from mne_nirs.experimental_design import create_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.experimental_design.tests.test_experimental_design import \
    _load_dataset


def test_run_GLM():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast
    design_matrix = create_first_level_design_matrix(raw_intensity,
                                                     drift_order=1,
                                                     drift_model='polynomial')
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    labels, glm_estimates = run_GLM(raw_haemo, design_matrix)

    assert len(labels) == len(raw_haemo.ch_names)

    # the estimates are nested. so cycle through to check correct number
    # are generated
    num = 0
    for est in glm_estimates:
        num += glm_estimates[est].theta.shape[1]
    assert num == len(raw_haemo.ch_names)
