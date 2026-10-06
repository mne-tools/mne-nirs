# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)

import os.path as op
from pathlib import Path

import mne_nirs

# Resolved at collection time from the real config (tests use a fake home)
data_path = mne_nirs.datasets.block_speech_noise.data_path(download=False)


def test_dataset_block_speech_noise():
    assert op.isdir(data_path)
    assert op.isdir(op.join(data_path, "sub-01"))

    # Accessing it again (with an explicit path) should not need a download
    datapath = mne_nirs.datasets.block_speech_noise.data_path(
        path=Path(data_path).parent, update_path=False, download=False
    )
    assert Path(datapath) == Path(data_path)
