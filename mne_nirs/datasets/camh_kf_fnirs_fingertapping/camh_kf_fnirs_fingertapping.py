# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs group tapping data. Once BIDS is finalised
# and the dataset complies I will add this to the MNE-Python library.

from functools import partial

import pooch
from mne.datasets import fetch_dataset
from mne.datasets.utils import has_dataset
from mne.utils import verbose

has_fnirs_motor_group_data = partial(has_dataset, name="camh_kf_fnirs_fingertapping")


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, verbose=None
):  # noqa: D103
    """
    CAMH KF fNIRS Fingertapping Dataset.

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
        If True, set the MNE_DATASETS_CAMHKFFNIRSFINGERTAPPING_PATH in
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
        archive_name="camh_kf_fnirs_fingertapping.zip",
        hash="md5:3f5e7566d262f9d636f6d7fc769ea1bf",
        url="https://osf.io/download/4nsrv?version=1",
        folder_name="camh_kf_fnirs_fingertapping",
        dataset_name="camh_kf_fnirs_fingertapping",
        config_key="MNE_DATASETS_CAMHKFFNIRSFINGERTAPPING_PATH",
    )

    return fetch_dataset(
        dataset_params,
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
        processor=pooch.Unzip(extract_dir=""),
    )
