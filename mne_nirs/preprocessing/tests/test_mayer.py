# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import pytest
from functools import partial
import numpy as np

from mne.utils._testing import requires_module
from mne_nirs.preprocessing import quantify_mayer_fooof


requires_fooof = partial(requires_module, name='fooof')


@requires_fooof
def test_mayer():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()

    raw_intensity = raw_intensity.pick(picks=range(8)).crop(tmax=600)

    with pytest.raises(RuntimeError, match="run on haemoglobin"):
        _ = quantify_mayer_fooof(raw_intensity)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    with pytest.raises(RuntimeError, match="run on haemoglobin"):
        _ = quantify_mayer_fooof(raw_od)

    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    df_mayer = quantify_mayer_fooof(raw_haemo)

    print(df_mayer)

    assert df_mayer.shape[0] == 2
    assert df_mayer.shape[1] == 4
    assert np.abs(df_mayer.query('Chromaphore == "hbo"'
                                 )["Centre Frequency"][0] - 0.1) < 0.05
