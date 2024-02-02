# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs audio or video speech dataset.

import os
import shutil
import pooch
from functools import partial

from mne.utils import verbose
from mne.datasets.utils import has_dataset
from mne.datasets import fetch_dataset

from ...fixes import _mne_path

has_block_speech_noise_data = partial(has_dataset,
                                      name='audio_or_visual_speech')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    """
    Audio and visual speech dataset with 8 participants.

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
        If True, set the MNE_DATASETS_FNIRSAUDIOVISUALSPEECH_PATH in
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
        archive_name='2021-fNIRS-Audio-visual-speech-'
                     'Broad-vs-restricted-regions.zip',
        hash='md5:16cac6565880dae6aed9b69100399d0b',
        url='https://osf.io/xwerv/download?version=1',
        folder_name='fNIRS-audio-visual-speech',
        dataset_name='audio_or_visual_speech',
        config_key='MNE_DATASETS_FNIRSAUDIOVISUALSPEECH_PATH',
    )

    dpath = fetch_dataset(dataset_params, path=path, force_update=force_update,
                          update_path=update_path, download=download,
                          processor=pooch.Unzip(
                              extract_dir="./fNIRS-audio-visual-speech"))
    dpath = str(dpath)

    # Do some wrangling to deal with nested directories
    bad_name = os.path.join(dpath, '2021-fNIRS-Audio-visual-speech-'
                                   'Broad-vs-restricted-regions')
    if os.path.isdir(bad_name):
        os.rename(bad_name, dpath + '.true')
        shutil.rmtree(dpath)
        os.rename(dpath + '.true', dpath)

    return _mne_path(dpath)
