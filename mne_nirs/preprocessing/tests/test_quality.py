# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne

from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed


def test_peak_power():

    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
    raw = mne.preprocessing.nirs.optical_density(raw)

    raw, scores, times = peak_power(raw)
    assert len(scores) == len(raw.ch_names)


def test_sci_windowed():

    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
    raw = mne.preprocessing.nirs.optical_density(raw)

    raw, scores, times = scalp_coupling_index_windowed(raw)
    assert len(scores) == len(raw.ch_names)
