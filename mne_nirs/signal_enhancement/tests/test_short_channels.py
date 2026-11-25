# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import mne
import numpy as np
import pytest

from mne_nirs.channels import get_long_channels, get_short_channels
from mne_nirs.signal_enhancement import short_channel_regression

fnirs_data_folder = mne.datasets.fnirs_motor.data_path(download=False)


def _load_dataset():
    """Load data and tidy it a bit"""
    fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()

    raw_intensity.crop(0, raw_intensity.annotations.onset[-1])

    new_des = [des for des in raw_intensity.annotations.description]
    new_des = ["A" if x == "1.0" else x for x in new_des]
    new_des = ["B" if x == "2.0" else x for x in new_des]
    new_des = ["C" if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(
        raw_intensity.annotations.onset, raw_intensity.annotations.duration, new_des
    )
    raw_intensity.set_annotations(annot)

    assert "fnirs_cw_amplitude" in raw_intensity
    assert len(np.unique(raw_intensity.annotations.description)) == 4

    return raw_intensity


def test_short():
    raw_intensity = _load_dataset()

    with pytest.raises(RuntimeError, match="must be optical density"):
        _ = short_channel_regression(raw_intensity)

    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    raw_od_corrected = short_channel_regression(raw_od)

    assert "fnirs_od" in raw_od_corrected

    with pytest.raises(RuntimeError, match="long channels present"):
        short_channel_regression(get_short_channels(raw_od))
    with pytest.raises(RuntimeError, match="short channels present"):
        short_channel_regression(get_long_channels(raw_od))


def test_short_channel_regression_matches_wavelength():
    """Test that each long channel is regressed on a short one of its wavelength."""
    rng = np.random.default_rng(0)
    n_times = 2000
    # independent systemic noise for each wavelength, shared by long and short
    systemic = dict(zip(("760", "850"), rng.standard_normal((2, n_times))))
    brain = 0.3 * rng.standard_normal(n_times)
    detector_x = dict(D1=0.03, D2=0.008)  # long and short separation (m)
    ch_names = ["S1_D1 760", "S1_D1 850", "S1_D2 760", "S1_D2 850"]
    data = np.array(
        [
            (brain if "D1" in name else 0.0) + systemic[name.split()[1]]
            for name in ch_names
        ]
    )
    info = mne.create_info(ch_names, 10.0, "fnirs_od")
    for ch in info["chs"]:
        x = detector_x[ch["ch_name"].split()[0].split("_")[1]]
        ch["loc"][3:6] = (0.0, 0.0, 0.0)  # source
        ch["loc"][6:9] = (x, 0.0, 0.0)  # detector
        ch["loc"][:3] = (x / 2, 0.0, 0.0)  # midpoint
        ch["loc"][9] = float(ch["ch_name"].split()[1])
    raw = mne.io.RawArray(data, info)

    corrected = short_channel_regression(raw)

    for name in ch_names[:2]:
        noise = systemic[name.split()[1]]
        assert abs(np.corrcoef(raw.get_data(name)[0], noise)[0, 1]) > 0.9
        assert abs(np.corrcoef(corrected.get_data(name)[0], noise)[0, 1]) < 0.05
        assert np.corrcoef(corrected.get_data(name)[0], brain)[0, 1] > 0.9

    with pytest.warns(RuntimeWarning, match="S1_D1 850.*850.0 nm"):
        short_channel_regression(raw.copy().drop_channels("S1_D2 850"))
