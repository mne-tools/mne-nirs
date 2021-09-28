# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs group tapping data. Once BIDS is finalised
# and the dataset complies I will add this to the MNE-Python library.

import os
import tempfile
import shutil
import pooch
from functools import partial

from mne.utils import verbose
from mne.datasets.utils import has_dataset
from mne.datasets import fetch_dataset

has_fnirs_motor_group_data = partial(has_dataset, name='fnirs_motor_group')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103

    dataset_params = dict(
        archive_name='BIDS-NIRS-Tapping-master.zip',
        hash='md5:d2b32a03601c9882aef534e22ad237ab',
        url='https://github.com/rob-luke/BIDS-NIRS-Tapping/archive/master.zip',
        folder_name='fNIRS-motor-group',
        dataset_name='fnirs_motor_group',
        config_key='MNE_DATASETS_FNIRSMOTORGROUP_PATH',
    )

    dpath = fetch_dataset(dataset_params, processor=pooch.Unzip(
        extract_dir="./fNIRS-motor-group"),
                         path=path,
                         force_update=force_update, update_path=update_path,
                         download=download)

    # Do some wrangling to deal with nested directories
    bad_name = os.path.join(dpath, 'BIDS-NIRS-Tapping-master')
    if os.path.isdir(bad_name):
        tmppath = tempfile.mkdtemp()
        os.rename(bad_name, tmppath)
        shutil.rmtree(dpath)
        os.rename(tmppath, dpath)

    return dpath
