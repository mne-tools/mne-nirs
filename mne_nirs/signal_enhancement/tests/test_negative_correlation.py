# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, BaseRaw, read_raw_fif
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.datasets import testing

from mne_nirs.signal_enhancement import enhance_negative_correlation


fname_nirx_15_0 = op.join(data_path(download=False),
                          'NIRx', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirx_15_2_recording_w_short')


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2,
                                    fname_nirx_15_0]))
def test_beer_lambert(fname, tmpdir):
    """Test converting NIRX files."""
    raw = read_raw_nirx(fname)
    with pytest.raises(RuntimeError, match='run on haemoglobin'):
        enhance_negative_correlation(raw)
    raw = optical_density(raw)
    with pytest.raises(RuntimeError, match='run on haemoglobin'):
        enhance_negative_correlation(raw)
    raw = beer_lambert_law(raw)
    assert 'hbo' in raw
    assert 'hbr' in raw
    raw_post = enhance_negative_correlation(raw)

    assert_almost_equal(np.corrcoef(raw_post._data[0],
                                    raw_post._data[1])[1, 0], -1)
    assert np.abs(np.corrcoef(raw_post._data[0], raw_post._data[3])[1, 0]) > 0

