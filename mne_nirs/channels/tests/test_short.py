# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import os
import mne
import numpy as np
import pytest

from mne.preprocessing.nirs import source_detector_distances
from mne_nirs.channels import get_long_channels, get_short_channels


def test_short_extraction():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()

    short_chans = get_short_channels(raw_intensity)

    original_num_channels = len(raw_intensity.ch_names)
    assert original_num_channels == 56

    short_num_channels = len(short_chans.ch_names)
    assert short_num_channels == 16  # There are 8 SDs in this set * hbo/hbr

    new_lens = source_detector_distances(short_chans.info)
    assert np.max(new_lens) <= 0.01

    # Now test for non standard short length
    short_chans = get_short_channels(raw_intensity, max_dist=0.052)

    short_num_channels = len(short_chans.ch_names)
    assert short_num_channels > 16  # There are 8 SDs in this set * hbo/hbr

    new_lens = source_detector_distances(short_chans.info)
    assert np.max(new_lens) <= 0.052

    # Check that we dont run on other types, eg eeg.
    raw_intensity.pick(picks=range(2))
    raw_intensity.set_channel_types({'S1_D1 760': 'eeg', 'S1_D1 850': 'eeg'},
                                    verbose='error')
    with pytest.raises(RuntimeError, match='NIRS signals only'):
        _ = get_short_channels(raw_intensity)


def test_long_extraction():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()

    long_chans = get_long_channels(raw_intensity)

    original_num_channels = len(raw_intensity.ch_names)
    assert original_num_channels == 56

    long_num_channels = len(long_chans.ch_names)
    assert long_num_channels == 56 - 16

    new_lens = source_detector_distances(long_chans.info)
    assert np.min(new_lens) >= 0.01

    # Now test for non standard short length
    long_chans = get_long_channels(raw_intensity, min_dist=0.022)

    long_num_channels = len(long_chans.ch_names)
    assert long_num_channels > 16  # There are 8 SDs in this set * hbo/hbr

    new_lens = source_detector_distances(long_chans.info)
    assert np.max(new_lens) >= 0.022

    # Check that we dont run on other types, eg eeg.
    raw_intensity.pick(picks=range(2))
    raw_intensity.set_channel_types({'S1_D1 760': 'eeg', 'S1_D1 850': 'eeg'},
                                    verbose='error')
    with pytest.raises(RuntimeError, match='NIRS signals only'):
        _ = get_long_channels(raw_intensity)
