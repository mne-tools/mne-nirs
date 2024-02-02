# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import os
from numpy.testing import assert_array_equal
import mne
from mne_nirs.channels import list_sources, list_detectors,\
    drop_sources, drop_detectors, pick_sources, pick_detectors


def _get_raw():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
    return raw


def test_list_sources():
    raw = _get_raw()

    sources = list_sources(raw)
    assert len(sources) == 8
    assert_array_equal(sources, range(1, 9))

    sources = list_sources(raw.copy().pick("S3_D2 760"))
    assert sources == [3]

    sources = list_sources(raw.copy().pick("S7_D15 850"))
    assert sources == [7]

    sources = list_sources(raw.copy().pick(["S7_D15 850", "S7_D15 760"]))
    assert sources == [7]

    sources = list_sources(raw.copy().pick(["S7_D15 850",
                                            "S3_D2 850", "S7_D15 760"]))
    assert_array_equal(sources, [3, 7])


def test_list_detectors():
    raw = _get_raw()

    detectors = list_detectors(raw)
    assert len(detectors) == 16
    assert_array_equal(detectors, range(1, 17))

    detectors = list_detectors(raw.copy().pick("S3_D2 760"))
    assert detectors == [2]

    detectors = list_detectors(raw.copy().pick("S7_D15 850"))
    assert detectors == [15]

    detectors = list_detectors(raw.copy().pick(["S7_D15 850",
                                                "S7_D15 760"]))
    assert detectors == [15]

    detectors = list_detectors(raw.copy().pick(["S7_D15 850",
                                                "S3_D2 850", "S7_D15 760"]))
    assert_array_equal(detectors, [2, 15])


def test_drop_sources():
    raw = _get_raw()

    sources = list_sources(raw)
    assert_array_equal(sources, range(1, 9))

    raw_dropped = drop_sources(raw.copy(), 3)
    assert 3 not in list_sources(raw_dropped)

    raw_dropped = drop_sources(raw.copy(), [7, 2])
    assert 7 not in list_sources(raw_dropped)
    assert 2 not in list_sources(raw_dropped)


def test_drop_detectors():
    raw = _get_raw()

    detectors = list_detectors(raw)
    assert_array_equal(detectors, range(1, 17))

    raw_dropped = drop_detectors(raw.copy(), 3)
    assert 3 not in list_detectors(raw_dropped)

    raw_dropped = drop_detectors(raw.copy(), [7, 2])
    assert 7 not in list_detectors(raw_dropped)
    assert 2 not in list_detectors(raw_dropped)

    raw_dropped = drop_detectors(raw.copy(), [1, 2])
    assert 12 in list_detectors(raw_dropped)
    assert 2 not in list_detectors(raw_dropped)
    assert 1 not in list_detectors(raw_dropped)


def test_pick_sources():
    raw = _get_raw()

    sources = list_sources(raw)
    assert_array_equal(sources, range(1, 9))

    raw_picked = pick_sources(raw.copy(), 3)
    assert 3 in list_sources(raw_picked)
    assert 2 not in list_sources(raw_picked)
    assert 1 not in list_sources(raw_picked)

    raw_picked = pick_sources(raw.copy(), [7, 2])
    assert 7 in list_sources(raw_picked)
    assert 2 in list_sources(raw_picked)
    assert 1 not in list_sources(raw_picked)
    assert 11 not in list_sources(raw_picked)
    assert 9 not in list_sources(raw_picked)


def test_pick_detectors():
    raw = _get_raw()

    sources = list_detectors(raw)
    assert_array_equal(sources, range(1, 17))

    raw_picked = pick_detectors(raw.copy(), 3)
    assert 3 in list_detectors(raw_picked)
    assert 2 not in list_detectors(raw_picked)
    assert 1 not in list_detectors(raw_picked)

    raw_picked = pick_detectors(raw.copy(), [7, 2])
    assert 7 in list_detectors(raw_picked)
    assert 2 in list_detectors(raw_picked)
    assert 1 not in list_detectors(raw_picked)
    assert 11 not in list_detectors(raw_picked)
    assert 9 not in list_detectors(raw_picked)
