# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)

import os.path as op
import mne_nirs


def test_dataset_av_speech():
    datapath = mne_nirs.datasets.audio_or_visual_speech.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "sub-01"))

    # First pass downloaded, check that second pass of access works
    datapath = mne_nirs.datasets.audio_or_visual_speech.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "sub-01"))
