# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op

import pandas as pd

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_nirx
from mne.transforms import apply_trans, _get_trans

from mne_nirs.io.fold._fold import _generate_all_locations,\
    _find_closest_standard_location, _read_fold_xls


# https://github.com/mne-tools/mne-testing-data/pull/72
fname_nirx_15_3_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout', 'nirx_15_3_recording')


def test_fold_reader():
    tbl = _read_fold_xls("./data/example.xls", atlas="Juelich")
    assert isinstance(tbl, pd.DataFrame)
    assert tbl.shape == (11, 10)
    assert "L Superior Frontal Gyrus" in \
           list(tbl["Landmark"])


@requires_testing_data
def test_label_finder():
    """Test locating labels."""
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')

    reference_locations = _generate_all_locations()

    # Test central head position source

    raw_cz = raw.copy().pick(25)
    source_locs = raw_cz.info['chs'][0]['loc'][3:6]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "Cz"

    # Test right auditory position detector

    raw_cz = raw.copy().pick(4)
    source_locs = raw_cz.info['chs'][0]['loc'][6:9]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "T8"

    # Test right auditory position source

    raw_cz = raw.copy().pick(4)
    source_locs = raw_cz.info['chs'][0]['loc'][3:6]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "TP8"

    # Test left auditory position source

    raw_cz = raw.copy().pick(1)
    source_locs = raw_cz.info['chs'][0]['loc'][3:6]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "T7"

    # Test left auditory position detector

    raw_cz = raw.copy().pick(1)
    source_locs = raw_cz.info['chs'][0]['loc'][6:9]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "TP7"

    # Test front position detector

    raw_cz = raw.copy().pick(14)
    source_locs = raw_cz.info['chs'][0]['loc'][6:9]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "AF2"

    # Test rear position detector

    raw_cz = raw.copy().pick(9)
    source_locs = raw_cz.info['chs'][0]['loc'][6:9]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(
        x, y, z, reference_locations) == "PO2"
