# Authors: Robert Luke <code@robertluke.net>
# License: BSD (3-clause)

import os.path as op

import mne_nirs


def test_dataset_snirf_aux():
    datapath = mne_nirs.datasets.snirf_with_aux.data_path()
    assert op.isdir(datapath)
    assert op.isfile(
        op.join(datapath, mne_nirs.datasets.snirf_with_aux.FNAME_SNIRF_AUX)
    )
    assert op.isfile(
        op.join(datapath, mne_nirs.datasets.snirf_with_aux.FNAME_SNIRF_AUX_NIRSPORT2)
    )
