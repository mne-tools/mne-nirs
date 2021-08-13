# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
import datetime
import h5py
from numpy.testing import assert_allclose
import pytest

from mne.datasets.testing import data_path, requires_testing_data
from mne.utils import requires_h5py, object_diff
from mne.io import read_raw_snirf, read_raw_nirx
from mne_nirs.io.snirf import write_raw_snirf, SPEC_FORMAT_VERSION


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
    test_file = tmpdir.join('test_raw.snirf')
    write_raw_snirf(raw_orig, test_file)
    raw = read_raw_snirf(test_file)

    # Check annotations are the same
    assert_allclose(raw.annotations.onset, raw_orig.annotations.onset)
    assert_allclose([float(d) for d in raw.annotations.description],
                    [float(d) for d in raw_orig.annotations.description])
    assert_allclose(raw.annotations.duration, raw_orig.annotations.duration)

    # Check data is the same
    assert_allclose(raw.get_data(), raw_orig.get_data())

    assert abs(raw_orig.info['meas_date'] - raw.info['meas_date']) < \
           datetime.timedelta(seconds=1)

    # Check info object is the same
    obj_diff = object_diff(raw.info, raw_orig.info)
    diffs = ''
    for line in obj_diff.splitlines():
        if ('logno' not in line) and \
                ('scanno' not in line) and \
                ('his_id' not in line) and\
                ('datetime mismatch' not in line):
            # logno and scanno are not used in processing
            diffs += f'\n{line}'
    assert diffs == ''

    _verify_snirf_required_fields(test_file)
    _verify_snirf_version_str(test_file)


@requires_h5py
@requires_testing_data
@pytest.mark.parametrize('fname', (
    fname_nirx_15_2,
))
def test_snirf_nobday(fname, tmpdir):
    """Ensure writing works when no birthday is present."""
    raw_orig = read_raw_nirx(fname, preload=True)
    raw_orig.info['subject_info'].pop('birthday', None)
    test_file = tmpdir.join('test_raw.snirf')
    write_raw_snirf(raw_orig, test_file)
    raw = read_raw_snirf(test_file)
    assert_allclose(raw.get_data(), raw_orig.get_data())


def _verify_snirf_required_fields(test_file):
    """Tests that all required fields are present.

    Uses Draft 3 of version 1.0 of the spec:
    https://github.com/fNIRS/snirf/blob/52de9a6724ddd0c9dcd36d8d11007895fed74205/snirf_specification.md
    """
    required_metadata_fields = [
        'SubjectID', 'MeasurementDate', 'MeasurementTime',
        'LengthUnit', 'TimeUnit', 'FrequencyUnit'
    ]
    required_measurement_list_fields = [
        'sourceIndex', 'detectorIndex', 'wavelengthIndex',
        'dataType', 'dataTypeIndex'
    ]

    with h5py.File(test_file, 'r') as h5:
        # Verify required base fields
        assert 'nirs' in h5
        assert 'formatVersion' in h5

        # Verify required metadata fields
        assert 'metaDataTags' in h5['/nirs']
        metadata = h5['/nirs/metaDataTags']
        for field in required_metadata_fields:
            assert field in metadata

        # Verify required data fields
        assert 'data1' in h5['/nirs']
        data1 = h5['/nirs/data1']
        assert 'dataTimeSeries' in data1
        assert 'time' in data1

        # Verify required fields for each measurementList
        measurement_lists = [k for k in data1.keys()
                             if k.startswith('measurementList')]
        for ml in measurement_lists:
            for field in required_measurement_list_fields:
                assert field in data1[ml]

        # Verify required fields for each stimulus
        stims = [k for k in h5['/nirs'].keys() if k.startswith('stim')]
        for stim in stims:
            assert 'name' in h5['/nirs'][stim]
            assert 'data' in h5['/nirs'][stim]

        # Verify probe fields
        assert 'probe' in h5['/nirs']
        probe = h5['/nirs/probe']
        assert 'wavelengths' in probe
        assert 'sourcePos3D' in probe or 'sourcePos2D' in probe
        assert 'detectorPos3D' in probe or 'detectorPos2D' in probe


def _verify_snirf_version_str(test_file):
    """Verify that the version string contains the correct spec version."""
    with h5py.File(test_file, 'r') as h5:
        version_str = h5['/formatVersion'][0].decode('UTF-8')
        expected_str = SPEC_FORMAT_VERSION
        assert version_str == expected_str
