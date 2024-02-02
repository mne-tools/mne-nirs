# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
import pytest

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.simulation import simulate_nirs_raw


def test_simulate_NIRS_single_channel():
    raw = simulate_nirs_raw(
        sfreq=3.0,
        amplitude=1.0,
        sig_dur=300.0,
        stim_dur=5.0,
        isi_min=15.0,
        isi_max=45.0,
    )
    assert "hbo" in raw
    assert raw.info["sfreq"] == 3.0
    assert raw.get_data().shape == (1, 900)
    assert np.max(raw.get_data()) < 1.2 * 1.0e-6
    assert raw.annotations.description[0] == "A"
    assert raw.annotations.duration[0] == 5
    assert np.min(np.diff(raw.annotations.onset)) > 15.0 + 5.0
    assert np.max(np.diff(raw.annotations.onset)) < 45.0 + 5.0

    with pytest.raises(AssertionError, match="Same number of"):
        _ = simulate_nirs_raw(
            sfreq=3.0,
            amplitude=[1.0, 2.0],
            sig_dur=300.0,
            stim_dur=5.0,
            isi_min=15.0,
            isi_max=45.0,
        )


def test_simulate_NIRS_multi_channel():
    raw = simulate_nirs_raw(
        sfreq=3.0,
        amplitude=[0.0, 2.0, 4.0],
        annot_desc=["Control", "Cond_A", "Cond_B"],
        stim_dur=[5, 5, 5],
        sig_dur=1500.0,
        isi_min=5.0,
        isi_max=15.0,
        hrf_model="spm",
    )

    design_matrix = make_first_level_design_matrix(
        raw, stim_dur=5.0, drift_order=0, drift_model="polynomial"
    )

    assert len(design_matrix["Control"]) == 1500 * 3
    assert len(design_matrix["Cond_A"]) == 1500 * 3

    # Make sure no extra channels. Specifically the default isn't present.
    with pytest.raises(KeyError, match="A"):
        len(design_matrix["A"])
