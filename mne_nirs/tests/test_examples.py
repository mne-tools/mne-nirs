# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

# This script runs each of the example scripts. It acts as a system test.

import os
import sys

import pytest


def examples_path():
    if not os.path.isdir("BIDS-NIRS-Tapping"):
        os.system(
            "git clone --depth 1 https://github.com/rob-luke/BIDS-NIRS-Tapping.git"
        )

    if os.path.isdir("examples"):
        path = "examples/general/"
    else:
        path = "../examples/general/"

    return path


@pytest.mark.filterwarnings("ignore:No bad channels to interpolate.*:")
@pytest.mark.filterwarnings("ignore:divide by zero encountered.*:")
@pytest.mark.filterwarnings("ignore:invalid value encountered.*:")
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Unstable on Windows")
@pytest.mark.examples
@pytest.mark.parametrize(
    "fname",
    (
        [
            "plot_01_data_io.py",
            "plot_05_datasets.py",
            "plot_10_hrf_simulation.py",
            "plot_11_hrf_measured.py",
            "plot_12_group_glm.py",
            "plot_13_fir_glm.py",
            "plot_14_glm_components.py",
            "plot_15_waveform.py",
            "plot_16_waveform_group.py",
            "plot_19_snirf.py",
            "plot_20_enhance.py",
            "plot_21_artifacts.py",
            "plot_22_quality.py",
            "plot_30_frequency.py",
            "plot_40_mayer.py",
            "plot_80_save_read_glm.py",
            "plot_99_bad.py",
        ]
    ),
)
def test_examples(fname, requires_pyvista):
    test_file_path = examples_path() + fname
    with open(test_file_path) as fid:
        code = fid.read()
    exec(code, locals(), locals())
