# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs group tapping data. Once BIDS is finalised
# and the dataset complies I will add this to the MNE-Python library.

import os
import os.path as op
from functools import partial
import shutil

from mne.utils import verbose
from mne.datasets.utils import (has_dataset)
from mne.datasets.utils import _get_path

has_fnirs_motor_group_data = partial(has_dataset, name='fnirs_motor_group')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    # TODO: all of the arguments to the function are being ignored!
    import pooch

    datapath = _get_path(None, 'MNE_DATASETS_SAMPLE_PATH', None)
    foldername = 'fNIRS-motor-group'
    archive_name = 'BIDS-NIRS-Tapping-master.zip'
    urls = {archive_name:
            'https://github.com/rob-luke/BIDS-NIRS-Tapping/archive/master.zip'}
    hashes = {archive_name: 'md5:8b1a09e4af64ec66426b9c70c07a5594'}

    fetcher = pooch.create(
        path=datapath,
        base_url='',    # Full URLs are given in the `urls` dict.
        version=None,   # Data versioning is decoupled from MNE-Python version.
        urls=urls,
        retry_if_failed=2,  # 2 retries = 3 total attempts
        registry=hashes
    )

    # fetch and unpack the data
    archive_name = 'BIDS-NIRS-Tapping-master.zip'
    downloader = pooch.HTTPDownloader(progressbar=True)
    unzip = pooch.Unzip(extract_dir=datapath)
    fetcher.fetch(fname=archive_name, downloader=downloader, processor=unzip)
    # after unpacking, remove the archive file
    os.remove(op.join(datapath, archive_name))
    return shutil.move(op.join(datapath, archive_name[:-4]),
                       op.join(datapath, foldername))
