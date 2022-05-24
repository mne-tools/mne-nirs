# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs group tapping data. Once BIDS is finalised
# and the dataset complies I will add this to the MNE-Python library.

import os
import shutil
import pooch
from functools import partial

from mne.utils import verbose
from mne.datasets.utils import has_dataset
from mne.datasets import fetch_dataset

from ...fixes import _mne_path

has_fnirs_motor_group_data = partial(has_dataset, name='fnirs_motor_group')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    """
    Motor task experiment data with 5 participants.

    Get path to local copy of data from the article :footcite:`shader2021use`.

    Parameters
    ----------
    path : None | str
        Location of where to look for the dataset.
        If None, the environment variable or config parameter is used.
        If it doesn’t exist, the “~/mne_data” directory is used.
        If the dataset is not found under the given path,
        the data will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_FNIRSMOTORGROUP_PATH in
        mne-python config to the given path. If None, the user is prompted.
    download : bool
        If False and the dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned
        as ‘’ (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    %(verbose)s

    Returns
    -------
    path : str
        Path to dataset directory.

    References
    ----------
    .. footbibliography::
    """

    dataset_params = dict(
        archive_name='BIDS-NIRS-Tapping-master.zip',
        hash='md5:da3cac7252005f0a64fdba5c683cf3dd',
        url='https://github.com/rob-luke/BIDS-NIRS-Tapping/archive/v0.1.0.zip',
        folder_name='fNIRS-motor-group',
        dataset_name='fnirs_motor_group',
        config_key='MNE_DATASETS_FNIRSMOTORGROUP_PATH',
    )

    dpath = fetch_dataset(dataset_params, path=path, force_update=force_update,
                          update_path=update_path, download=download,
                          processor=pooch.Unzip(
                              extract_dir="./fNIRS-motor-group"))
    dpath = str(dpath)

    # Do some wrangling to deal with nested directories
    bad_name = os.path.join(dpath, 'BIDS-NIRS-Tapping-0.1.0')
    if os.path.isdir(bad_name):
        os.rename(bad_name, dpath + '.true')
        shutil.rmtree(dpath)
        os.rename(dpath + '.true', dpath)

    return _mne_path(dpath)
