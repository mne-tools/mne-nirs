# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
from numpy.testing import assert_allclose
import pytest

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_nirx
from mne.transforms import apply_trans, _get_trans

from mne_nirs.io.fold._fold import _generate_all_locations, _find_closest_standard_location


# https://github.com/mne-tools/mne-testing-data/pull/72
fname_nirx_15_3_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout', 'nirx_15_3_recording')


@requires_testing_data
def test_label_finder():
    """Test locating labels."""
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')

    reference_locations = _generate_all_locations(system=["1010"])

    # Test central head position source

    raw_cz = raw.copy().pick(25)
    raw_cz.plot_sensors()
    source_locs = raw_cz.info['chs'][0]['loc'][3:6]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(x, y, z, reference_locations) == "Cz"

    # Test right auditory position detector

    raw_cz = raw.copy().pick(4)
    raw_cz.plot_sensors()
    source_locs = raw_cz.info['chs'][0]['loc'][6:9]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(x, y, z, reference_locations) == "T8"

    # Test left auditory position source

    raw_cz = raw.copy().pick(4)
    raw_cz.plot_sensors()
    source_locs = raw_cz.info['chs'][0]['loc'][6:9]

    mni_locs = apply_trans(head_mri_t, source_locs)
    x = mni_locs[0]
    y = mni_locs[1]
    z = mni_locs[2]
    assert _find_closest_standard_location(x, y, z, reference_locations) == "T8"


