# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import pytest
from mne_nirs.channels import picks_pair_to_idx


def test_roi_picks():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()

    picks = picks_pair_to_idx(raw, [[1, 1], [1, 2], [5, 13], [8, 16]])

    assert raw.ch_names[picks[0]] == "S1_D1 760"
    assert raw.ch_names[picks[1]] == "S1_D1 850"

    assert raw.ch_names[picks[2]] == "S1_D2 760"
    assert raw.ch_names[picks[3]] == "S1_D2 850"

    assert raw.ch_names[picks[4]] == "S5_D13 760"
    assert raw.ch_names[picks[5]] == "S5_D13 850"

    assert raw.ch_names[picks[6]] == "S8_D16 760"
    assert raw.ch_names[picks[7]] == "S8_D16 850"

    # Test what happens when a pair that doesn't exist is requested (15-13)
    with pytest.raises(ValueError, match='No matching'):
        picks_pair_to_idx(raw, [[1, 1], [1, 2], [15, 13], [8, 16]])

    with pytest.warns(RuntimeWarning, match='No matching channels'):
        picks = picks_pair_to_idx(raw, [[1, 1], [1, 2], [15, 13], [8, 16]],
                                  on_missing='warning')
    assert len(picks) == 6  # Missing should be ignored

    picks = picks_pair_to_idx(raw, [[1, 1], [1, 2], [15, 13], [8, 16]],
                              on_missing='ignore')
    assert len(picks) == 6

    # Test usage for ROI downstream functions
    group_by = dict(Left_ROI=picks_pair_to_idx(raw, [[1, 1], [1, 2], [5, 13]]),
                    Right_ROI=picks_pair_to_idx(raw, [[3, 3], [3, 11]]))
    assert group_by['Left_ROI'] == [0, 1, 2, 3, 34, 35]
    assert group_by['Right_ROI'] == [18, 19, 20, 21]

    # Ensure we dont match [1, 1] to S1_D11
    # Check easy condition
    picks = picks_pair_to_idx(raw, [[1, 1]])
    assert picks == [0, 1]
    # Force in tricky situation
    raw.info["ch_names"][2] = 'S1_D11 760'
    raw.info["ch_names"][3] = 'S1_D11 850'
    picks = picks_pair_to_idx(raw, [[1, 1]])
    assert picks == [0, 1]

    picks = picks_pair_to_idx(raw, [[21, 91], [91, 2]], on_missing='ignore')
    assert picks == []
