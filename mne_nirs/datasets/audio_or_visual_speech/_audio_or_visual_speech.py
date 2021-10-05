# Authors: Robert Luke <mail@robertluke.net>
# License: BSD (3-clause)
#
# This downloads my fnirs audio or video speech dataset.

import os
import tempfile
import shutil
import pooch
from functools import partial

from mne.utils import verbose
from mne.datasets.utils import has_dataset
from mne.datasets import fetch_dataset

has_block_speech_noise_data = partial(has_dataset,
                                      name='audio_or_visual_speech')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    """
    Systemic correction regression based on nearest short channel.

    Method as described by NIRx and based on
    :footcite:`fabbri2004optical`, :footcite:`saager2005direct`,
    and :footcite:`scholkmann2014measuring`.

    Parameters
    ----------
    path : instance of Raw
        Raw instance containing optical density data.
    force_update : number
        Channels less than this distance are considered short (m).
    update_path : number
        Channels less than this distance are considered short (m).
    download : number
        Channels less than this distance are considered short (m).
    verbose : number
        Channels less than this distance are considered short (m).

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.

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

    # Do some wrangling to deal with nested directories
    bad_name = os.path.join(dpath, '2021-fNIRS-Audio-visual-speech-'
                                   'Broad-vs-restricted-regions')
    if os.path.isdir(bad_name):
        tmppath = tempfile.mkdtemp()
        os.rename(bad_name, tmppath)
        shutil.rmtree(dpath)
        os.rename(tmppath, dpath)

    return dpath
