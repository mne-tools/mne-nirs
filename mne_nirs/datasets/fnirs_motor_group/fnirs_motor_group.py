# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs group tapping data. Once BIDS is finalised
# and the dataset complies I will add this to the MNE-Python library.

import os.path as op
from functools import partial

from mne.utils import verbose
from mne.datasets.utils import (has_dataset)
from mne.datasets.utils import _get_path, _download, _extract


has_fnirs_motor_group_data = partial(has_dataset, name='fnirs_motor_group')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    datapath = _get_path(None, 'MNE_DATASETS_SAMPLE_PATH', None)

    downloadpath = 'https://github.com/rob-luke/' \
                   'BIDS-NIRS-Tapping/archive/master.zip'

    if not op.isdir(datapath + "/MNE-fNIRS-motor-group-data/"):
        remove_archive, full = _download(datapath, downloadpath,
                                         "MNE-fNIRS-motor-group-data.zip",
                                         "ef2d41fc2824ded78d065214cdbc9b36")
        _extract(datapath, "fNIRS-motor-group",
                 op.join(datapath, "MNE-fNIRS-motor-group-data"),
                 full, op.join(datapath, "BIDS-NIRS-Tapping-master"), True)

    datapath = op.join(datapath, "MNE-fNIRS-motor-group-data/")

    return datapath
