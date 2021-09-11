# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import os
from numpy.testing import assert_array_equal
import mne
from mne_nirs.channels import list_sources, list_detectors


def test_list_sources():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()

    sources = list_sources(raw)
    assert len(sources) == 8

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
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()

    detectors = list_detectors(raw)
    assert len(detectors) == 16

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
