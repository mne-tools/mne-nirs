import os
import pytest
import mne

from mne_nirs.preprocessing import peak_power
from mne_nirs.visualisation import plot_timechannel_quality_metric


@pytest.mark.filterwarnings('ignore:.*nilearn.glm module is experimental.*:')
def test_peak_power():

    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
    raw = mne.preprocessing.nirs.optical_density(raw)

    raw, scores, times = peak_power(raw)
    assert len(scores) == len(raw.ch_names)

    fig = plot_timechannel_quality_metric(raw, scores, times)
    assert len(fig.axes) == 4
