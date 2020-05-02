# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne_nirs


def test_simulate_block_design():
    raw = mne_nirs.simulation.simulate_block_design()
    assert 'hbo' in raw
