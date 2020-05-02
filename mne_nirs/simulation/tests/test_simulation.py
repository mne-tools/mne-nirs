# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne_nirs


def test_simulate_NIRS():
    raw = mne_nirs.simulation.simulate_nirs_raw()
    assert 'hbo' in raw
