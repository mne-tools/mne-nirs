# Authors: Robert Luke <code@robertluke.net>
# License: BSD (3-clause)

import os.path as op
from pathlib import Path

import mne_nirs

# Resolved at collection time from the real config (tests use a fake home)
fname = mne_nirs.datasets.snirf_with_aux.data_path(download=False)


def test_dataset_snirf_aux():
    assert op.isfile(fname)
    assert "2022-08-05_002.snirf" in str(fname)

    # Accessing it again (with an explicit path) should not need a download
    datapath = mne_nirs.datasets.snirf_with_aux.data_path(
        path=Path(fname).parents[1], update_path=False, download=False
    )
    assert Path(datapath) == Path(fname)
