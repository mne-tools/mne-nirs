# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from mne_nirs.simulation import simulate_nirs_raw
import numpy as np
import pytest
from mne_nirs.experimental_design import make_first_level_design_matrix


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_simulate_NIRS_single_channel():

    raw = simulate_nirs_raw(sfreq=3., amplitude=1., sig_dur=300., stim_dur=5.,
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
        _ = simulate_nirs_raw(sfreq=3., amplitude=[1., 2.], sig_dur=300.,
                              stim_dur=5., isi_min=15., isi_max=45.)


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_simulate_NIRS_multi_channel():

    raw = simulate_nirs_raw(sfreq=3.,
                            amplitude=[0., 2., 4.],
                            annot_desc=['Control',
                                        'Cond_A',
                                        'Cond_B'],
                            stim_dur=[5, 5, 5],
                            sig_dur=1500.,
                            isi_min=5., isi_max=15.,
                            hrf_model='spm')

    design_matrix = make_first_level_design_matrix(raw, stim_dur=5.0,
                                                   drift_order=0,
                                                   drift_model='polynomial')

    assert len(design_matrix['Control']) == 1500 * 3
    assert len(design_matrix['Cond_A']) == 1500 * 3

    # Make sure no extra channels. Specifically the default isn't present.
    with pytest.raises(KeyError, match='A'):
        len(design_matrix['A'])
