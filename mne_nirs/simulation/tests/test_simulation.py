# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from mne_nirs.simulation import simulate_nirs_raw
import numpy as np


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
