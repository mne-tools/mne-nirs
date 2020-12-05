# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import numpy as np
import pytest

from mne_nirs.signal_enhancement import short_channel_regression
from mne_nirs.channels import get_long_channels, get_short_channels


def _load_dataset():
    """Load data and tidy it a bit"""
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()

    raw_intensity.crop(0, raw_intensity.annotations.onset[-1])

    new_des = [des for des in raw_intensity.annotations.description]
    new_des = ['A' if x == "1.0" else x for x in new_des]
    new_des = ['B' if x == "2.0" else x for x in new_des]
    new_des = ['C' if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(raw_intensity.annotations.onset,
                            raw_intensity.annotations.duration, new_des)
    raw_intensity.set_annotations(annot)

    assert 'fnirs_cw_amplitude' in raw_intensity
    assert len(np.unique(raw_intensity.annotations.description)) == 4

    return raw_intensity


def test_short():
    raw_intensity = _load_dataset()

    with pytest.raises(RuntimeError, match="must be optical density"):
        _ = short_channel_regression(raw_intensity)

    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    raw_od_corrected = short_channel_regression(raw_od)

    assert 'fnirs_od' in raw_od_corrected

    with pytest.raises(RuntimeError, match="long channels present"):
        short_channel_regression(get_short_channels(raw_od))
    with pytest.raises(RuntimeError, match="short channels present"):
        short_channel_regression(get_long_channels(raw_od))
