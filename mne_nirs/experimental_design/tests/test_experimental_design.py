# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import mne_nirs


def test_create_boxcar():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()
    raw_intensity = raw_intensity.pick(picks=[0])
    mne_nirs.experimental_design.create_boxcar(raw_intensity)


def test_create_hrf():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()
    raw_intensity = raw_intensity.pick(picks=[0])
    mne_nirs.experimental_design.create_hrf(raw_intensity)
