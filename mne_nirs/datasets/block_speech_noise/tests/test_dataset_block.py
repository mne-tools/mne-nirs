# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)

import os.path as op
import mne_nirs


def test_dataset_block_speech_noise():
    datapath = mne_nirs.datasets.block_speech_noise.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "sub-17"))

    # First pass downloaded, check that second pass of access works
    datapath = mne_nirs.datasets.block_speech_noise.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "sub-06"))
