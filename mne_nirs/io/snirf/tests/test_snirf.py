# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
import datetime
import h5py
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import pandas as pd
from snirf import validateSnirf, Snirf

from mne.datasets.testing import data_path, requires_testing_data
from mne.utils import requires_h5py, object_diff
from mne.io import read_raw_snirf, read_raw_nirx
from mne.preprocessing.nirs import optical_density
from mne_nirs.io.snirf import write_raw_snirf, SPEC_FORMAT_VERSION, \
    read_snirf_aux_data
import mne_nirs.datasets.snirf_with_aux as aux
from mne.utils import check_version


fname_nirx_15_0 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout',
                                'nirx_15_2_recording_w_short')
fname_snirf_aux = aux.data_path()

pytest.importorskip('mne', '1.0')  # these tests are broken on 0.24!

requires_mne_1_4 = pytest.mark.skipif(not check_version('mne', '1.4'),
                                      reason='Needs MNE-Python 1.4')


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

    result = validateSnirf(str(test_file))
    if result.is_valid():
        result.display()
    assert result.is_valid()

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
                ('dig' not in line) and\
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


@requires_h5py
@requires_testing_data
@pytest.mark.parametrize('fname', (
    fname_nirx_15_2,
))
def test_snirf_extra_atlasviewer(fname, tmpdir):
    """Ensure writing atlasviewer landmarks."""
    raw_orig = read_raw_nirx(fname, preload=True)
    test_file = tmpdir.join('test_raw.snirf')

    write_raw_snirf(raw_orig, test_file, add_montage=False)
    raw = read_raw_snirf(test_file)
    assert len([i['ident'] for i in raw.info['dig']]) == 35

    write_raw_snirf(raw_orig, test_file, add_montage=True)
    raw = read_raw_snirf(test_file)
    assert len([i['ident'] for i in raw.info['dig']]) == 129
    snirf = Snirf(str(test_file), "r")
    assert len(snirf.nirs[0].probe.landmarkLabels) == 129
    assert "Fpz" in snirf.nirs[0].probe.landmarkLabels
    assert "P7" in snirf.nirs[0].probe.landmarkLabels
    assert "HP_20" in snirf.nirs[0].probe.landmarkLabels


def _verify_snirf_required_fields(test_file):
    """Tests that all required fields are present.

    Uses version 1.1 of the spec:
    https://raw.githubusercontent.com/fNIRS/snirf/v1.1/snirf_specification.md
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
        version_str = h5['/formatVersion'][()].decode('UTF-8')
        expected_str = SPEC_FORMAT_VERSION
        assert version_str == expected_str


def test_aux_read():
    """Test reading auxiliary data from SNIRF file."""
    raw = read_raw_snirf(fname_snirf_aux)
    a = read_snirf_aux_data(fname_snirf_aux, raw)
    assert type(a) is pd.DataFrame
    assert 'accelerometer_2_z' in a
    assert len(a['gyroscope_1_z']) == len(raw.times)


@requires_h5py
@requires_testing_data
@pytest.mark.parametrize('fname', (
    fname_nirx_15_2,
    fname_nirx_15_2_short,
))
def test_snirf_stim_roundtrip(fname, tmpdir):
    """Ensure snirf annotations are written."""
    raw_orig = read_raw_nirx(fname, preload=True)
    assert raw_orig.annotations.duration[0] == 1
    raw_mod = raw_orig.copy()
    test_file = tmpdir.join('test_raw_no_mod.snirf')
    write_raw_snirf(raw_mod, test_file)
    raw = read_raw_snirf(test_file)
    assert_array_equal(raw_orig.annotations.onset,
                       raw.annotations.onset)
    assert_array_equal(raw_orig.annotations.duration,
                       raw.annotations.duration)
    assert_array_equal(raw_orig.annotations.description,
                       raw.annotations.description)


@requires_mne_1_4
@requires_h5py
@requires_testing_data
@pytest.mark.parametrize('fname', (
    fname_nirx_15_2,
    fname_nirx_15_2_short,
))
@pytest.mark.parametrize('newduration', (
    1, 2, 3
))
def test_snirf_duration(fname, newduration, tmpdir):
    """Ensure snirf annotations are written to file."""
    raw_orig = read_raw_nirx(fname, preload=True)
    assert raw_orig.annotations.duration[0] == 1
    raw_mod = raw_orig.copy()
    raw_mod.annotations.set_durations(newduration)
    test_file = tmpdir.join('test_raw_duration.snirf')
    write_raw_snirf(raw_mod, test_file)
    raw = read_raw_snirf(test_file)
    assert raw.annotations.duration[0] == newduration
    assert raw.annotations.duration[-1] == newduration


@requires_h5py
@requires_testing_data
@pytest.mark.parametrize('fname', (
    fname_nirx_15_2,
    fname_nirx_15_2_short,
))
def test_optical_density_roundtrip(fname, tmpdir):
    """Ensure snirf annotations are written."""
    raw_nirx = read_raw_nirx(fname, preload=True)
    od_orig = optical_density(raw_nirx)
    assert od_orig.annotations.duration[0] == 1
    test_file = tmpdir.join('test_raw_no_mod.snirf')
    write_raw_snirf(od_orig, test_file)
    od = read_raw_snirf(test_file)
    assert 'fnirs_od' in od
    assert_array_equal(od_orig.annotations.onset,
                       od.annotations.onset)
    assert_array_equal(od_orig.annotations.duration,
                       od.annotations.duration)
    assert_array_equal(od_orig.annotations.description,
                       od.annotations.description)
