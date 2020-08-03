# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
from numpy.testing import assert_allclose

from mne.datasets.testing import data_path, requires_testing_data
from mne.utils import requires_h5py, object_diff
from mne.io import read_raw_snirf, read_raw_nirx
from mne_nirs.io.snirf import write_raw_snirf
from mne.transforms import apply_trans, _get_trans

fname_original = op.join(data_path(download=False),
                         'NIRx', 'nirx_15_2_recording_w_short')


@requires_testing_data
@requires_h5py
def test_snirf_write(tmpdir):
    """Test reading NIRX files."""
    raw_orig = read_raw_nirx(fname_original, preload=True)
    write_raw_snirf(raw_orig, tmpdir.join('test_raw.snirf'))
    raw = read_raw_snirf(tmpdir.join('test_raw.snirf'))

    # Copied straight from NIRX tests
    allowed_dist_error = 0.0002
    locs = [ch['loc'][6:9] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][3:5] == 'D1'
    assert_allclose(
        mni_locs[0], [-0.0841, -0.0464, -0.0129], atol=allowed_dist_error)

    assert raw.info['ch_names'][4][3:5] == 'D3'
    assert_allclose(
        mni_locs[4], [0.0846, -0.0142, -0.0156], atol=allowed_dist_error)

    assert raw.info['ch_names'][8][3:5] == 'D2'
    assert_allclose(
        mni_locs[8], [0.0207, -0.1062, 0.0484], atol=allowed_dist_error)

    assert raw.info['ch_names'][12][3:5] == 'D4'
    assert_allclose(
        mni_locs[12], [-0.0196, 0.0821, 0.0275], atol=allowed_dist_error)

    assert raw.info['ch_names'][16][3:5] == 'D5'
    assert_allclose(
        mni_locs[16], [-0.0360, 0.0276, 0.0778], atol=allowed_dist_error)

    assert raw.info['ch_names'][19][3:5] == 'D6'
    assert_allclose(
        mni_locs[19], [0.0352, 0.0283, 0.0780], atol=allowed_dist_error)

    assert raw.info['ch_names'][21][3:5] == 'D7'
    assert_allclose(
        mni_locs[21], [0.0388, -0.0477, 0.0932], atol=allowed_dist_error)

    # Check annotations are the same
    assert_allclose(raw.annotations.onset, raw_orig.annotations.onset)
    assert_allclose([float(d) for d in raw.annotations.description],
                    [float(d) for d in raw_orig.annotations.description])
    assert_allclose(raw.annotations.duration, raw_orig.annotations.duration)

    # Check names are the same
    assert raw.info['ch_names'] == raw_orig.info['ch_names']

    # Check frequencies are the same
    num_chans = len(raw.ch_names)
    new_chs = raw.info['chs']
    ori_chs = raw_orig.info['chs']
    assert_allclose([new_chs[idx]['loc'][9] for idx in range(num_chans)],
                    [ori_chs[idx]['loc'][9] for idx in range(num_chans)])

    # # Check data is the same
    assert_allclose(raw.get_data(), raw_orig.get_data())

    # Check info object is the same
    obj_diff = object_diff(raw.info, raw_orig.info)
    num_diffs = 0
    for line in obj_diff.splitlines():
        if ('logno' not in line) and ('scanno' not in line):
            num_diffs += 1
            print(line)
    # TODO: Set to zero before requesting merge
    assert num_diffs == 0
