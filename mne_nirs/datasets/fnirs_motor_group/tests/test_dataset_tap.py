# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)

import os.path as op
import mne_nirs


def test_dataset_tapping_group():
    datapath = mne_nirs.datasets.fnirs_motor_group.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "sub-01"))
