# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne
import mne_nirs
import numpy as np

from mne.utils import (requires_pysurfer, traits_test)

mne.viz.set_3d_backend('pyvista')


@requires_pysurfer
@traits_test
def test_plot_nirs_source_detector():
    data_path = mne.datasets.testing.data_path()
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    raw = mne.io.read_raw_nirx(data_path + '/NIRx/nirx_15_2_recording_w_short')

    mne_nirs.visualisation.plot_nirs_source_detector(
        np.random.randn(len(raw.ch_names)),
        raw.info, show_axes=True,
        subject='fsaverage',
        trans='fsaverage',
        surfaces=['brain'],
        fnirs=False,
        subjects_dir=subjects_dir,
        verbose=True)
