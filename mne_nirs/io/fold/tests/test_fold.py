# -*- coding: utf-8 -*-
# Author: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from pathlib import Path
from shutil import copyfile

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import mne
from mne.channels import make_standard_montage
from mne.channels.montage import transform_to_head
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_nirx, read_fiducials
from mne.utils import check_version

from mne_nirs.io.fold._fold import _generate_montage_locations,\
    _find_closest_standard_location, _read_fold_xls
from mne_nirs.io import fold_landmark_specificity
from mne_nirs.io.fold import fold_channel_specificity

thisfile = Path(__file__).parent.resolve()
foldfile = thisfile / "data" / "example.xls"

# https://github.com/mne-tools/mne-testing-data/pull/72
fname_nirx_15_3_short = Path(data_path(download=False)) / \
    'NIRx' / 'nirscout' / 'nirx_15_3_recording'

requires_xlrd = pytest.mark.skipif(
    not check_version('xlrd', '1.0'), reason='Requires xlrd >= 1.0')


@requires_xlrd
@pytest.mark.parametrize('fold_files', (str, None, list))
def test_channel_specificity(monkeypatch, tmp_path, fold_files):
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)
    raw.pick(range(2))
    kwargs = dict()
    n_want = 6
    if fold_files is list:
        kwargs = dict(fold_files=[foldfile])
    elif fold_files is str:
        kwargs = dict(fold_files=tmp_path)
        n_want *= 2
    else:
        assert fold_files is None
        monkeypatch.setenv('MNE_NIRS_FOLD_PATH', str(tmp_path))
        assert len(kwargs) == 0
        with pytest.raises(FileNotFoundError, match=r'fold_files\[0\] does.*'):
            fold_channel_specificity(raw)
        n_want *= 2
    copyfile(foldfile, tmp_path / '10-10.xls')
    copyfile(foldfile, tmp_path / '10-5.xls')
    res = fold_channel_specificity(raw, **kwargs)
    assert len(res) == 2
    assert res[0].shape == (n_want, 14)
    montage = make_standard_montage(
        'standard_1005', head_size=0.09700884729534559)
    fids = read_fiducials(
        Path(mne.__file__).parent / 'data' / 'fsaverage' /
        'fsaverage-fiducials.fif')[0]
    for f in fids:
        f['coord_frame'] = montage.dig[0]['coord_frame']
    montage.dig[:3] = fids
    S, D = raw.ch_names[0].split()[0].split('_')
    assert S == 'S1' and D == 'D2'
    montage.rename_channels({'PO8': S, 'P6': D})  # not in the tables!
    # taken from standard_1020.elc
    s_mri = np.array([55.6666, -97.6251, 2.7300]) / 1000.
    d_mri = np.array([67.8877, -75.9043, 28.0910]) / 1000.
    trans = mne.transforms._get_trans('fsaverage', 'mri', 'head')[0]
    ch_pos = montage.get_positions()['ch_pos']
    assert_allclose(ch_pos[S], s_mri, atol=1e-6)
    assert_allclose(ch_pos[D], d_mri, atol=1e-6)
    raw.set_montage(montage)
    montage = transform_to_head(montage)
    s_head = mne.transforms.apply_trans(trans, s_mri)
    d_head = mne.transforms.apply_trans(trans, d_mri)
    assert_allclose(montage._get_ch_pos()['S1'], s_head, atol=1e-6)
    assert_allclose(montage._get_ch_pos()['D2'], d_head, atol=1e-6)
    for ch in raw.info['chs']:
        assert_allclose(ch['loc'][3:6], s_head, atol=1e-6)
        assert_allclose(ch['loc'][6:9], d_head, atol=1e-6)
    res_1 = fold_channel_specificity(raw, **kwargs)[0]
    assert res_1.shape == (0, 14)
    # TODO: This is wrong, should be P08 not P08h, and distance should be 0 mm!
    with pytest.warns(RuntimeWarning, match='.*PO8h?/P6.*TP8/T8.*'):
        res_1 = fold_channel_specificity(raw, interpolate=True, **kwargs)[0]
    montage.rename_channels({S: D, D: S})  # reversed
    with pytest.warns(RuntimeWarning, match='.*PO8h?/P6.*TP8/T8.*'):
        res_2 = fold_channel_specificity(raw, interpolate=True, **kwargs)[0]
    # We should check the whole thing, but this is probably good enough
    assert (res_1['Specificity'] == res_2['Specificity']).all()


@requires_xlrd
def test_landmark_specificity():
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)
    with pytest.warns(RuntimeWarning, match='No fOLD table entry'):
        res = fold_landmark_specificity(raw, "L Superior Frontal Gyrus",
                                        [foldfile], interpolate=True)
    assert len(res) == len(raw.ch_names)
    assert np.max(res) <= 100
    assert np.min(res) >= 0


@requires_xlrd
def test_fold_workflow():
    # Read raw data
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)
    reference_locations = _generate_montage_locations()
    channel_of_interest = raw.copy().pick(1)

    # Get source and detector labels
    source_locs = channel_of_interest.info['chs'][0]['loc'][3:6]
    source_label = _find_closest_standard_location(source_locs,
                                                   reference_locations)
    assert source_label == "T7"

    detector_locs = channel_of_interest.info['chs'][0]['loc'][6:9]
    detector_label = _find_closest_standard_location(detector_locs,
                                                     reference_locations)
    assert detector_label == "TP7"

    # Find correct fOLD elements
    tbl = _read_fold_xls(foldfile, atlas="Juelich")
    tbl = tbl.query("Source == @source_label").\
        query("Detector == @detector_label")

    # Query region of interest
    specificity = tbl.query("Landmark == 'L Mid Orbital Gyrus'")["Specificity"]
    assert specificity.values == 12.34


@requires_xlrd
def test_fold_reader():
    tbl = _read_fold_xls(foldfile, atlas="Juelich")
    assert isinstance(tbl, pd.DataFrame)
    assert tbl.shape == (11, 10)
    assert "L Superior Frontal Gyrus" in \
           list(tbl["Landmark"])


@requires_testing_data
def test_label_finder():
    """Test locating labels."""
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)
    reference_locations = _generate_montage_locations()

    # Test central head position source
    raw_tmp = raw.copy().pick(25)
    assert _find_closest_standard_location(
        raw_tmp.info['chs'][0]['loc'][3:6],
        reference_locations) == "Cz"

    # Test right auditory position detector
    raw_tmp = raw.copy().pick(4)
    assert _find_closest_standard_location(
        raw_tmp.info['chs'][0]['loc'][6:9],
        reference_locations) == "T8"

    # Test right auditory position source
    raw_tmp = raw.copy().pick(4)
    assert _find_closest_standard_location(
        raw_tmp.info['chs'][0]['loc'][3:6],
        reference_locations) == "TP8"

    # Test left auditory position source
    raw_tmp = raw.copy().pick(1)
    assert _find_closest_standard_location(
        raw_tmp.info['chs'][0]['loc'][3:6],
        reference_locations) == "T7"

    # Test left auditory position detector
    raw_tmp = raw.copy().pick(1)
    assert _find_closest_standard_location(
        raw_tmp.info['chs'][0]['loc'][6:9],
        reference_locations) == "TP7"
