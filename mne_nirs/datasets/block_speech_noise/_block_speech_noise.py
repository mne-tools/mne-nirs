# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs speech and noise dataset.

import os
import tempfile
import shutil
import pooch
from functools import partial

from mne.utils import verbose
from mne.datasets.utils import has_dataset
from mne.datasets import fetch_dataset

has_block_speech_noise_data = partial(has_dataset, name='block_speech_noise')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103

    dataset_params = dict(
        archive_name='2021-fNIRS-Analysis-Methods-Passive-Auditory.zip',
        hash='md5:569c0fbafa575e344e90698c808dfdd3',
        url='https://osf.io/bjfu7/download?version=1',
        folder_name='fNIRS-block-speech-noise',
        dataset_name='block_speech_noise',
        config_key='MNE_DATASETS_FNIRSSPEECHNOISE_PATH',
    )

    dpath = fetch_dataset(dataset_params, path=path, force_update=force_update,
                          update_path=update_path, download=download,
                          processor=pooch.Unzip(
                              extract_dir="./fNIRS-block-speech-noise"))

    # Do some wrangling to deal with nested directories
    bad_name = os.path.join(dpath, '2021-fNIRS-Analysis-Methods-'
                                   'Passive-Auditory')
    if os.path.isdir(bad_name):
        tmppath = tempfile.mkdtemp()
        os.rename(bad_name, tmppath)
        shutil.rmtree(dpath)
        os.rename(tmppath, dpath)

    return dpath
