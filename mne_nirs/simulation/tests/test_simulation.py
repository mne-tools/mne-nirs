# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from mne_nirs.simulation import simulate_nirs_raw
from mne_nirs.statistics import run_GLM
import numpy as np
import pytest
from mne_nirs.utils._io import glm_to_tidy, _tidy_long_to_wide
from mne_nirs.experimental_design import make_first_level_design_matrix


def test_simulate_NIRS():

    raw = simulate_nirs_raw(sfreq=3., amplitude=1.,
                            sig_dur=300., stim_dur=5.,
                            isi_min=15., isi_max=45.)
    assert 'hbo' in raw
    assert raw.info['sfreq'] == 3.
    assert raw.get_data().shape == (1, 900)
    assert np.max(raw.get_data()) < 1.2 * 1.e-6
    assert raw.annotations.description[0] == 'A'
    assert raw.annotations.duration[0] == 5
    assert np.min(np.diff(raw.annotations.onset)) > 15. + 5.
    assert np.max(np.diff(raw.annotations.onset)) < 45. + 5.

    with pytest.raises(AssertionError, match='Same number of'):
        raw = simulate_nirs_raw(sfreq=3., amplitude=[1., 2.],
                                sig_dur=300., stim_dur=5.,
                                isi_min=15., isi_max=45.)

    raw = simulate_nirs_raw(sfreq=3.,
                            amplitude=[0., 2., 4.],
                            annot_desc=['Control',
                                        'Cond_A',
                                        'Cond_B'],
                            stim_dur=[5, 5, 5],
                            sig_dur=900.,
                            isi_min=15., isi_max=45.)
    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.0,
                                                   drift_order=1,
                                                   drift_model='polynomial')
    glm_est = run_GLM(raw, design_matrix)
    df = glm_to_tidy(raw, glm_est, design_matrix)
    df = _tidy_long_to_wide(df)

    assert df.query("condition in ['Control']")['theta'].values[0] == \
        pytest.approx(0)
    assert df.query("condition in ['Cond_A']")['theta'].values[0] == \
        pytest.approx(2e-6)
    assert df.query("condition in ['Cond_B']")['theta'].values[0] == \
        pytest.approx(4e-6)
