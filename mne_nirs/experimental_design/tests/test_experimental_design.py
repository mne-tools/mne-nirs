# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import mne
import numpy as np
import pytest
from mne.utils import catch_logging
from numpy.testing import assert_allclose

import mne_nirs
from mne_nirs.experimental_design import (
    drift_high_pass,
    longest_inter_annotation_interval,
    make_first_level_design_matrix,
)
from mne_nirs.experimental_design._experimental_design import (
    _design_matrix_vif,
    _vif_lstsq,
)
from mne_nirs.simulation import simulate_nirs_raw


def _load_dataset():
    """Load data and tidy it a bit"""
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()

    raw_intensity.crop(0, raw_intensity.annotations.onset[-1])

    new_des = [des for des in raw_intensity.annotations.description]
    new_des = ["A" if x == "1.0" else x for x in new_des]
    new_des = ["B" if x == "2.0" else x for x in new_des]
    new_des = ["C" if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(
        raw_intensity.annotations.onset, raw_intensity.annotations.duration, new_des
    )
    raw_intensity.set_annotations(annot)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )
    raw_intensity.pick(picks[dists > 0.01])

    assert "fnirs_cw_amplitude" in raw_intensity
    assert len(np.unique(raw_intensity.annotations.description)) == 4

    return raw_intensity


def test_create_boxcar():
    raw_intensity = _load_dataset()
    raw_intensity = raw_intensity.pick(picks=[0])  # Keep the test fast
    bc = mne_nirs.experimental_design.create_boxcar(raw_intensity)

    assert bc.shape[0] == raw_intensity._data.shape[1]
    assert bc.shape[1] == len(np.unique(raw_intensity.annotations.description))

    assert np.max(bc) == 1
    assert np.min(bc) == 0

    # The value of the boxcar should be 1 when a trigger fires
    assert (
        bc[int(raw_intensity.annotations.onset[0] * raw_intensity.info["sfreq"]), :][0]
        == 1
    )

    # Only one condition was ever present at a time in this data
    # So boxcar should never overlap across channels
    assert np.max(np.mean(bc, axis=1)) * bc.shape[1] == 1


def test_create_design():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast
    design_matrix = make_first_level_design_matrix(
        raw_intensity, drift_order=1, drift_model="polynomial"
    )

    assert design_matrix.shape[0] == raw_intensity._data.shape[1]
    # Number of columns is number of conditions plus the drift plus constant
    assert (
        design_matrix.shape[1]
        == len(np.unique(raw_intensity.annotations.description)) + 2
    )


def test_cropped_raw():
    # Ensure timing is correct for cropped signals
    raw = simulate_nirs_raw(
        sfreq=1.0,
        amplitude=1.0,
        sig_dur=300.0,
        stim_dur=1.0,
        isi_min=20.0,
        isi_max=40.0,
    )

    onsets = raw.annotations.onset
    onsets_after_crop = [onsets[idx] for idx in np.where(onsets > 100)]

    raw.crop(tmin=100)
    design_matrix = make_first_level_design_matrix(
        raw, drift_order=0, drift_model="polynomial"
    )

    # 100 corrects for the crop time above
    # 4 is peak time after onset
    new_idx = np.round(onsets_after_crop[0][0]) - 100 + 4
    assert design_matrix["A"][new_idx] > 0.09


def test_high_pass_helpers():
    # Test the helpers give reasonable values
    raw = simulate_nirs_raw(
        sfreq=1.0,
        amplitude=1.0,
        sig_dur=300.0,
        stim_dur=1.0,
        isi_min=20.0,
        isi_max=38.0,
    )
    lisi, names = longest_inter_annotation_interval(raw)
    lisi = lisi[0]
    assert lisi >= 20
    assert lisi <= 40
    assert drift_high_pass(raw) >= 1 / (40 * 2)
    assert drift_high_pass(raw) <= 1 / (20 * 2)


def test_design_matrix_vif_statsmodels():
    """Test that our VIF matches the statsmodels implementation."""
    statsmodels = pytest.importorskip("statsmodels.stats.outliers_influence")

    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast
    kwargs = dict(drift_order=1, drift_model="polynomial")
    design_matrix = make_first_level_design_matrix(raw_intensity, **kwargs)
    assert "constant" in design_matrix.columns

    # return_vif=True returns the same design matrix plus the VIF
    design_matrix_2, vif = make_first_level_design_matrix(
        raw_intensity, return_vif=True, **kwargs
    )
    assert_allclose(design_matrix_2.values, design_matrix.values)
    assert list(vif.index) == [
        name for name in design_matrix.columns if name != "constant"
    ]
    # VIF is by construction at least 1
    assert (vif >= 1).all()

    # statsmodels wants the intercept to be part of the matrix it is given
    want = np.array(
        [
            statsmodels.variance_inflation_factor(design_matrix.values, ii)
            for ii, name in enumerate(design_matrix.columns)
            if name != "constant"
        ]
    )
    assert_allclose(vif.values, want, rtol=1e-7)


def test_design_matrix_vif_collinear():
    """Test VIF detection of collinear regressors."""
    from pandas import DataFrame

    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, 3))
    design_matrix = DataFrame(
        dict(
            a=data[:, 0],
            b=data[:, 1],
            # c is a noisy copy of b, so both should have an elevated VIF
            c=data[:, 1] + 0.1 * data[:, 2],
            constant=np.ones(500),
        )
    )
    with catch_logging(verbose=True) as log:
        vif = _design_matrix_vif(design_matrix)
    log = log.getvalue()
    assert "High collinearity" in log
    assert "constant" not in vif.index
    assert 1 <= vif["a"] < 1.1
    assert vif["b"] > 5
    assert vif["c"] > 5

    # An exactly duplicated regressor is perfectly collinear
    design_matrix["c"] = design_matrix["b"]
    vif = _design_matrix_vif(design_matrix)
    assert np.isinf(vif["b"])
    assert np.isinf(vif["c"])

    # As is a regressor that duplicates the intercept
    design_matrix["c"] = 1.0
    assert np.isinf(_design_matrix_vif(design_matrix)["c"])

    # Orthogonal regressors have a VIF of exactly 1
    design_matrix = DataFrame(dict(a=[1.0, -1.0, 1.0, -1.0], b=[1.0, 1.0, -1.0, -1.0]))
    assert_allclose(_design_matrix_vif(design_matrix).values, [1.0, 1.0])


def test_design_matrix_vif_lstsq():
    """Test the least squares fallback against the Cholesky fast path."""
    from pandas import DataFrame

    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 5))
    data[:, 4] = data[:, :4] @ [1.0, 2.0, 3.0, 4.0] + 0.01 * data[:, 4]
    design_matrix = DataFrame({str(ii): data[:, ii] for ii in range(5)})
    want = _design_matrix_vif(design_matrix)
    assert want.max() > 5  # otherwise this is not a useful comparison

    data = data - data.mean(0)
    data /= np.linalg.norm(data, axis=0)
    assert_allclose(_vif_lstsq(data), want.values, rtol=1e-6)


def test_design_matrix_vif_degenerate():
    """Test VIF of design matrices with no usable regressors."""
    from pandas import DataFrame

    # Nothing but an intercept
    vif = _design_matrix_vif(DataFrame(dict(constant=np.ones(10))))
    assert len(vif) == 0

    # Regressors that are all intercepts
    vif = _design_matrix_vif(DataFrame(dict(a=np.ones(10), b=2 * np.ones(10))))
    assert_allclose(vif.values, [np.inf, np.inf])


def test_design_matrix_vif_logging():
    """Test that VIF logging can be controlled."""
    raw = simulate_nirs_raw(sfreq=3.0, sig_dur=200.0, stim_dur=5.0)
    # VIF is reported in the log even when it is not returned
    with catch_logging(verbose=True) as log:
        dm = make_first_level_design_matrix(raw)
    assert not isinstance(dm, tuple)
    assert "Maximum design matrix VIF" in log.getvalue()
    with catch_logging(verbose="error") as log:
        make_first_level_design_matrix(raw, verbose="error")
    assert log.getvalue() == ""

    # return_vif and verbose are keyword-only
    with pytest.raises(TypeError, match="positional arguments"):
        make_first_level_design_matrix(
            raw, 1.0, "glover", "cosine", 0.01, 1, (0,), None, None, -24, 50, True
        )
