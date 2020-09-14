# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
import datetime
from numpy.testing import assert_allclose
import pytest

from mne.datasets.testing import data_path, requires_testing_data
from mne.utils import requires_h5py, object_diff
from mne.io import read_raw_snirf, read_raw_nirx
from mne_nirs.io.snirf import write_raw_snirf


fname_nirx_15_0 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout',
                                'nirx_15_2_recording_w_short')


@requires_h5py
@requires_testing_data
@pytest.mark.parametrize('fname', (
    fname_nirx_15_2_short,
    fname_nirx_15_2,
    fname_nirx_15_0
))
def test_snirf_write(fname, tmpdir):
    """Test reading NIRX files."""
    raw_orig = read_raw_nirx(fname, preload=True)
    write_raw_snirf(raw_orig, tmpdir.join('test_raw.snirf'))
    raw = read_raw_snirf(tmpdir.join('test_raw.snirf'))

    # Check annotations are the same
    assert_allclose(raw.annotations.onset, raw_orig.annotations.onset)
    assert_allclose([float(d) for d in raw.annotations.description],
                    [float(d) for d in raw_orig.annotations.description])
    assert_allclose(raw.annotations.duration, raw_orig.annotations.duration)

    # Check data is the same
    assert_allclose(raw.get_data(), raw_orig.get_data())

    assert abs(raw_orig.info["meas_date"] - raw.info["meas_date"]) < \
           datetime.timedelta(seconds=1)

    # Check info object is the same
    obj_diff = object_diff(raw.info, raw_orig.info)
    diffs = ''
    for line in obj_diff.splitlines():
        if ('logno' not in line) and ('scanno' not in line) and\
                ('datetime mismatch' not in line):
            # logno and scanno are not used in processing
            diffs += f'\n{line}'
    assert diffs == ''
