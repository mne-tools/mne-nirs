# Authors: Robert Luke <code@robertluke.net>
# License: BSD (3-clause)
#
# This downloads SNIRF data that includes auxiliary channels.

import os
from functools import partial

import pooch
from mne.datasets import fetch_dataset
from mne.datasets.utils import has_dataset
from mne.utils import verbose

from ...fixes import _mne_path

has_fnirs_snirf_aux_data = partial(has_dataset, name="snirf_with_aux")


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, verbose=None
):  # noqa: D103
    """
    SNIRF file with auxiliary channels.

    Get path to local copy of data.

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
        If True, set the MNE_DATASETS_SNIRFAUX_PATH in
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
    """
    dataset_params = dict(
        archive_name="2022-08-05_002.snirf.zip",
        hash="md5:c012485b80d762a65b05d812761c0f4c",
        url="https://osf.io/download/682ddabd4d28c665accacc7c?version=2",
        folder_name="fNIRS-SNIRF-aux",
        dataset_name="snirf_with_aux",
        config_key="MNE_DATASETS_SNIRFAUX_PATH",
    )

    dpath = fetch_dataset(
        dataset_params,
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
        processor=pooch.Unzip(extract_dir="./fNIRS-SNIRF-aux"),
    )
    dpath = str(dpath)

    return _mne_path(os.path.join(dpath, "2022-08-05_002.snirf"))
