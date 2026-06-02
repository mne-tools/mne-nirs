# License: BSD (3-clause)

import os
from pathlib import Path

import pooch
from mne.utils import verbose


@verbose
def data_path(path=None, force_update=False, download=True, verbose=None):  # noqa: D103
    """
    OpenNeuro Kernel Flow 2 Audio Dataset (ds006545).

    Downloads a single subject/session SNIRF file from the OpenNeuro
    Reliability-Dubois2024 dataset (TD-fNIRS moments, auditory task).

    Parameters
    ----------
    path : None | str
        Location of where to look for the dataset.
        If None, the "~/mne_data" directory is used.
    force_update : bool
        Force re-download of the file even if a local copy exists.
    download : bool
        If False and the file has not been downloaded yet,
        it will not be downloaded and the path will be returned
        as '' (empty string).
    %(verbose)s

    Returns
    -------
    path : str
        Path to the downloaded SNIRF file.

    References
    ----------
    Dubois, J., et al. (2024). Reliability of brain metrics derived from a
    Time-Domain Functional Near-Infrared Spectroscopy System. Scientific
    Reports, 14(1), 17500. https://doi.org/10.1038/s41598-024-68555-9
    """
    if path is None:
        path = os.path.join(os.path.expanduser("~"), "mne_data")
    dest = Path(path) / "openneuro_kf2_audio"

    fname = "sub-bed8fefe_ses-1_task-audio_nirs.snirf"
    fpath = dest / fname

    if not download and not fpath.exists():
        return ""

    dest.mkdir(parents=True, exist_ok=True)
    url = (
        "https://s3.amazonaws.com/openneuro.org/ds006545/"
        "sub-bed8fefe/ses-1/nirs/"
        "sub-bed8fefe_ses-1_task-audio_nirs.snirf"
    )

    pooch.retrieve(
        url=url,
        known_hash=None,
        fname=fname,
        path=str(dest),
    )

    return str(fpath)
