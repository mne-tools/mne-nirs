# Authors: Robert Luke <code@robertluke.net>
# License: BSD (3-clause)

import os.path as op
import mne_nirs


def test_dataset_snirf_aux():
    datapath = mne_nirs.datasets.snirf_with_aux.data_path()
    assert op.isfile(datapath)
    assert "2022-08-05_002.snirf" in str(datapath)
