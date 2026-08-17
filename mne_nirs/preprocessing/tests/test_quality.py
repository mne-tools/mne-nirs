# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from pathlib import Path

import mne  # type: ignore
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed


@pytest.fixture(name="fnirs_motor_data")
def fixture_fnirs_motor_data() -> mne.io.BaseRaw:
    """Read and return motor experiment data."""
    fnirs_data_folder = Path(mne.datasets.fnirs_motor.data_path())
    fnirs_raw_dir = fnirs_data_folder / "Participant-1"
    raw = mne.io.read_raw_nirx(str(fnirs_raw_dir), verbose=True).load_data()
    return mne.preprocessing.nirs.optical_density(raw)


@pytest.fixture(name="fnirs_labnirs_3wl_data")
def fixture_fnirs_labnirs_3wl_data() -> mne.io.BaseRaw:
    """Read and return 3-wavelength testing data."""
    if not mne.datasets.has_dataset("testing"):
        pytest.skip("Requires testing dataset")
    fname_labnirs_3wl = (
        mne.datasets.testing.data_path(download=False)
        / "SNIRF"
        / "Labnirs"
        / "labnirs_3wl_raw_recording.snirf"
    )
    raw = mne.io.read_raw_snirf(fname_labnirs_3wl)
    ch_names = [
        "S2_D2 780",
        "S2_D2 805",
        "S2_D2 830",
        "S1_D2 780",
        "S1_D2 805",
        "S1_D2 830",
        "S1_D1 780",
        "S1_D1 805",
        "S1_D1 830",
    ]
    assert raw.ch_names[:9] == ch_names
    return mne.preprocessing.nirs.optical_density(raw)


def find_annotations(
    raw: mne.io.BaseRaw,
    description: str,
    windows: list[int],
    channel_names: list[str],
    window_time: float,
) -> np.ndarray:
    """Return a matrix of whether expected annotations are found in the expected places.

    Parameters
    ----------
    raw : instance of Raw
        The Raw data, annotated (e.g. with BAD_SCI or BAD_PeakPower annotations)
    description : str
        The description of the annotation to look for (e.g. "BAD_SCI").
    windows : list of int
        The window numbers (starting from 0) to look for annotations in.
    channel_names : list of str
        The channel names to look for annotations in.
    window_time : float
        The duration of the window in seconds (e.g. 10), used to calculate the
        expected annotation onset times.

    Returns
    -------
    marks : array (n_channels, n_windows)
        A boolean array where True indicates that an annotation with the specified
        description was found for the corresponding channel and window.
    """
    marks = np.zeros((len(channel_names), len(windows)), dtype=bool)

    for ann in raw.annotations:
        # skip if not expected label
        if ann["description"] != description:
            continue

        # find corresponding window
        ann_window = round(ann["onset"] / window_time)  # type: ignore
        try:
            col = windows.index(ann_window)
        except ValueError:
            continue

        # confirm channel name
        for ann_name in ann["ch_names"]:  # type: ignore
            if ann_name in channel_names:
                marks[channel_names.index(ann_name), col] = True

    return marks


@pytest.fixture(
    name="fnirs_dataset",
    params=["fnirs_motor_data", "fnirs_labnirs_3wl_data"],
)
def fixture_fnirs_dataset(request) -> mne.io.BaseRaw:
    """Each dataset becomes a separate parametrized test instance."""
    return request.getfixturevalue(request.param)


def test_peak_power_runs(fnirs_dataset: mne.io.BaseRaw) -> None:
    """Test that `peak_power` successfully runs with test data."""
    _, scores, _ = peak_power(fnirs_dataset.copy())
    assert len(scores) == len(fnirs_dataset.ch_names)


def test_sci_windowed_runs(fnirs_dataset: mne.io.BaseRaw) -> None:
    """Test that `scalp_coupling_index_windowed` successfully runs with test data."""
    _, scores, _ = scalp_coupling_index_windowed(fnirs_dataset.copy())
    assert len(scores) == len(fnirs_dataset.ch_names)


def test_sci_windowed_known_values(fnirs_motor_data: mne.io.BaseRaw):
    """Test segmented SCI with known correlation values for 2-wavelength data.

    Three channel pairs are overwritten with synthetic data that produce
    predictable scores. First, the channels are overwritten with the same
    signal to achieve perfect correlation (and keep the number of bad marks
    low). Then, various cases are tested in two, distant, 30-second windows
    (235 samples at 7.8125 Hz):

    - Pair 0 (ch 0-1): both = signal (SCI ≈ +1) throughout
    - Pair 1 (ch 2-3): W1 scale-invariant (SCI ≈ +1), W2 ch1 = -signal (SCI ≈ -1).
    - Pair 2 (ch 4-5): W1 noise + signal (SCI low), W2 noise + noise (SCI ≈ 0).

    The scores, annotations, and whether they're associated with the correct
    windows are confirmed.

    For now, the test uses optical density data, even though SCI was meant
    to be calculated on raw voltage measurement. In the future, when the
    ``scalp_coupling_index_windowed`` method is updated to use raw data,
    this test will need to be updated as well.
    """
    raw = fnirs_motor_data.copy()

    sfreq = raw.info["sfreq"]
    time_window = 30
    num_channels = 6
    windowA = 5
    windowB = 11

    rng = np.random.default_rng(seed=123456)

    # time_window=30 -> window_samples=235, n_windows=98
    w_numsamples = int(np.ceil(time_window * sfreq))
    # window A start:end with 0.5 window buffer around it for filter leak
    wA = round((windowA - 0.5) * w_numsamples), round((windowA + 1.5) * w_numsamples)
    # windows B start:end with buffer
    wB = round((windowB - 0.5) * w_numsamples), round((windowB + 1.5) * w_numsamples)

    # write the same signal in all the used channels
    signal = np.sin(2.0 * np.pi * np.arange(raw.n_times) / sfreq) - 0.5
    raw._data[0:num_channels] = signal[np.newaxis, :].copy()

    # Pair 0 (ch 0, 1)
    # signal in both, SCI≈1 throughout and no annotations in either window

    # Pair 1 (ch 2, 3):
    # window A: scale invariant (SCI ≈ +1) vs base signal in channel 3
    raw._data[2][wA[0] : wA[1]] = 0.3 * signal[wA[0] : wA[1]] + 2
    # window B: anti-correlation (SCI ≈ -1) vs base signal in channel 2
    raw._data[3][wB[0] : wB[1]] = -signal[wB[0] : wB[1]]

    # Pair 2 (ch 4, 5):
    # window A: noisy signal (low SCI) vs clean signal in channel 5
    raw._data[4][wA[0] : wA[1]] = (
        rng.normal(size=(wA[1] - wA[0],)) * 3.0 + signal[wA[0] : wA[1]]
    ) / 4
    # window B: both noise (SCI ≈ 0)
    raw._data[4][wB[0] : wB[1]] = rng.normal(size=(wB[1] - wB[0],))
    raw._data[5][wB[0] : wB[1]] = rng.random(size=(wB[1] - wB[0],)) - 0.5

    # calculate SCI quality
    raw, scores, times_out = scalp_coupling_index_windowed(
        raw, time_window=time_window, threshold=0.7
    )

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 98)
    assert len(times_out) == 98

    # verify SCI scores
    assert_allclose(scores[0:2, windowA], [1, 1], atol=0.05)  # pair 0
    assert_allclose(scores[0:2, windowB], [1, 1], atol=0.05)
    assert_allclose(scores[2:4, windowA], [1, 1], atol=0.05)  # pair 1, window A
    assert_allclose(scores[2:4, windowB], [-1, -1], atol=0.05)  # pair 1, window B
    assert_allclose(scores[4:6, windowA], [0.44, 0.44], atol=0.05)  # pair 2, window A
    assert_allclose(
        scores[4:6, windowB], [-0.0471, -0.0471], atol=0.01
    )  # pair 2, window B

    # verify that BAD_SCI annotations exist for expected channels/windows
    marks = find_annotations(
        raw, "BAD_SCI", [windowA, windowB], raw.ch_names[:num_channels], time_window
    )

    expected = np.array([False, False] * 2 + [False, True] * 2 + [True, True] * 2)
    assert_array_equal(marks.ravel(), expected)


def test_sci_windowed_known_values_multi_wavelength(
    fnirs_labnirs_3wl_data: mne.io.BaseRaw,
) -> None:
    """Test segmented SCI with known correlation values for >=3-wavelength data.

    This test focuses on multi-wavelength specific cases that were not covered
    in the 2-wavelength test, using the labnirs 3-wavelength SNIRF recording
    (250 samples at 19.6 Hz).

    - Group 1 (ch 0-2): all the same signal (SCI ≈ +1)
    - Group 2 (ch 3-5): one channel has inverted signal (SCI ≈ -1)
    - Group 3 (ch 6-8): one channel has noisy signal, another just noise
      (smallest SCI counts, SCI will be very small)
    - Group 4 (ch 9-11): same as group 3, different order (same SCI)

    For now, the test uses optical density data, even though SCI was meant
    to be calculated on raw voltage measurement. In the future, when the
    ``scalp_coupling_index_windowed`` method is updated to use raw data,
    this test will need to be updated as well.
    """
    raw = fnirs_labnirs_3wl_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 4
    num_channels = 12

    rng = np.random.default_rng(seed=123456)

    # write the same signal in all the used channels
    signal = np.sin(2 * np.pi * 1.0 * np.arange(raw.n_times) / sfreq) - 0.5
    # no need to use copy() for signal as the data is not modified later on
    # if part of the data is overwritten (as in the 2-wl test), copy() is needed
    raw._data[0:num_channels] = signal[np.newaxis, :]

    # Group 1 (ch 0, 1, 2): all perfectly correlated

    # Group 2 (ch 3, 4, 5): one channel has inverted signal
    raw._data[4] = -signal

    # Group 3 (ch 6, 7, 8): one channel is noisy, another just noise
    rand1 = rng.random((raw.n_times,)) - 0.5
    raw._data[6] = 0.1  # SCI = 0.09 if changed alone
    raw._data[8] = rand1  #  SCI = -0.149 if alone

    # Group 4 (ch 9, 10, 11): same as group 3 just different order
    raw._data[10] = rand1
    raw._data[11] = 0.1

    # calculate SCI quality
    raw, scores, times_out = scalp_coupling_index_windowed(
        raw, time_window=time_window, threshold=0.7
    )

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 3)
    assert len(times_out) == 3

    # verify SCI scores (in the 2nd, middle window)
    assert_allclose(scores[0:3, 1], 1, atol=0.05)  # group 1
    assert_allclose(scores[3:6, 1], -1, atol=0.05)  # group 2
    assert_allclose(scores[6:9, 1], -0.149, atol=0.01)  # group 3
    assert_allclose(scores[9:12, 1], scores[6:9, 1], atol=0.01)  # group 4

    # verify that BAD_SCI annotations exist for expected channels
    marks = find_annotations(
        raw, "BAD_SCI", [1], raw.ch_names[:num_channels], time_window
    )

    expected = np.repeat([False, True, True, True], 3)
    assert_array_equal(marks.ravel(), expected)


def test_peak_power_known_values(fnirs_motor_data: mne.io.BaseRaw) -> None:
    """Test segmented PP with known spectral properties for 2-wavelength data.

    First, test channels are overwritten with a sinusoid wave. Then, test data
    with known PP scores is written in two, distant, 30-second windows in the
    same test channels (time_window=30 s, 235 samples/window at
    7.8125 Hz):

    - Pair 0 (ch 0-1): W1 both sinusoid (high PP), W2 scale (invariant, high)
    - Pair 1 (ch 2-3): W1 phase shift, W2 inverted (invariant, both high PP)
    - Pair 2 (ch 4-5): W1 other frequences (high PP), W2 noisy (lower PP)
    - Pair 3 (ch 6-7): W1 only noise, W2 only other frequencies (both PP ≈ 0)

    The scores, annotations, and whether they're associated with the correct
    windows are confirmed.

    For now, the test uses optical density data, even though PP was meant
    to be calculated on raw voltage measurement. In the future, when the
    tested method is updated to use raw data, this test will need to be
    updated as well.
    """
    raw = fnirs_motor_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 30
    num_channels = 8
    windowA = 5
    windowB = 11

    rng = np.random.default_rng(seed=123456)

    # time_window=30 -> window_samples=235, n_windows=98
    w_numsamples = int(np.ceil(time_window * sfreq))
    # window A start:end with 0.5 window buffer around it for filter leak
    wA = round((windowA - 0.5) * w_numsamples), round((windowA + 1.5) * w_numsamples)
    # windows B start:end with buffer
    wB = round((windowB - 0.5) * w_numsamples), round((windowB + 1.5) * w_numsamples)

    t = np.arange(raw.n_times) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) - 0.5  # base "heartbeat", 1 Hz
    lf_signal = np.sin(2 * np.pi * 0.2 * t) - 0.5  # low-freq, 0.2 Hz
    hf_signal = np.sin(2 * np.pi * 2.0 * t) - 0.5  # high-freq, 2 Hz

    # write base sinusoid signal into all test channels
    raw._data[0:num_channels] = signal[np.newaxis, :].copy()

    # Pair 0 (ch 0, 1):
    # window A: base signals (perfect PP)
    # window B: scale invariant (perfect PP)
    # test data in one channel is tested against base signal in the other
    raw._data[1][wB[0] : wB[1]] = 0.3 * signal[wB[0] : wB[1]] + 2

    # Pair 1 (ch 2, 3):
    # window A: phase invariant (perfect PP)
    raw._data[2][wA[0] : wA[1]] = signal[wA[0] + 2 : wA[1] + 2]
    # window B: inverted signal (same as phase shift, perfect PP)
    raw._data[3][wB[0] : wB[1]] = -signal[wB[0] : wB[1]]

    # Pair 2 (ch 4, 5):
    # window A: other freqs are filtered out (perfect PP)
    raw._data[4][wA[0] : wA[1]] = (signal + lf_signal + hf_signal)[wA[0] : wA[1]]
    # window B: noisy signal (lower PP)
    raw._data[5][wB[0] : wB[1]] = (
        rng.normal(size=(wB[1] - wB[0],)) * 2 + signal[wB[0] : wB[1]]
    )

    # Pair 3 (ch 6, 7):
    # window A: only noise (PP ≈ 0)
    raw._data[6][wA[0] : wA[1]] = rng.normal(size=(wB[1] - wB[0],))
    # window B: only other frequencies (PP ≈ 0)
    raw._data[7][wB[0] : wB[1]] = lf_signal[wB[0] : wB[1]] + hf_signal[wB[0] : wB[1]]

    # calculate PP quality
    raw, scores, times_out = peak_power(raw, time_window=time_window, threshold=0.1)

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 98)
    assert len(times_out) == 98

    # verify scores
    assert_allclose(scores[0:4, [windowA, windowB]], 10, atol=0.05)  # pairs 0 and 1
    assert_allclose(scores[4:6, windowA], [10, 10], atol=0.05)  # pair 2, window A
    assert_allclose(scores[4:6, windowB], [3.5, 3.5], atol=0.05)  # pair 2, window B
    assert_allclose(scores[6:8, windowA], [0, 0], atol=0.05)  # pair 3, window A
    assert_allclose(scores[6:8, windowB], [0, 0], atol=0.05)  # pair 3, window B

    # verify that BAD_PeakPower annotations exist for expected channels/windows
    marks = find_annotations(
        raw,
        "BAD_PeakPower",
        [windowA, windowB],
        raw.ch_names[:num_channels],
        time_window,
    )

    expected = np.array([False, False] * 6 + [True, True] * 2)
    assert_array_equal(marks.ravel(), expected)


def test_peak_power_known_values_multi_wavelength(
    fnirs_labnirs_3wl_data: mne.io.BaseRaw,
) -> None:
    """Test segmented PP with known spectral properties for >=3-wavelength data.

    This test focuses on multi-wavelength specific cases that were not covered
    in the 2-wavelength test, using the labnirs 3-wavelength SNIRF recording
    (250 samples at 19.6 Hz). The recording is long enough for three 4-second
    windows, of which the tests are evaluated in the middle one.

    - Group 1 (ch 0-2): in-band sinusoid throughout (perfect PP)
    - Group 2 (ch 3-5): signal + noisy signal + noise (smallest wins, PP≈0)
    - Group 3 (ch 6-8): group 2 with different order (same PP)
    """
    raw = fnirs_labnirs_3wl_data.copy().load_data()

    sfreq = raw.info["sfreq"]
    time_window = 4
    num_channels = 9
    rng = np.random.default_rng(seed=123456)
    t = np.arange(raw.n_times) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t)  # base "heartbeat", 1 Hz
    noheart = np.sin(2 * np.pi * 0.1 * t) + np.sin(
        2 * np.pi * 3 * t
    )  # 0.1 Hz + 3 Hz, not in the 0.5-2.5 Hz PP band
    noisy = signal + rng.normal(size=(raw.n_times,)) * 2

    # write the same signal in all the used channels
    # No need to use copy() for signal as the data is not modified later on.
    # If part of the data were overwritten (as in the 2-wl test),
    # copy() would be needed.
    raw._data[0:num_channels] = signal[np.newaxis, :]

    # Group 1 (S2-D2): all perfectly correlated, no need to change

    # Group 2 (S1-D2):
    # base signal, noisy signal, no heart signal, should have very low PP
    raw._data[4] = noisy.copy()
    raw._data[5] = noheart.copy()

    # Group 3 (S1-D1): group 2 reordered, should have same-ish PP
    # It won't be exactly the same value as the result of the correlation
    # will be reversed if the compared channels are reversed, which results
    # in minimally different periodograms.
    raw._data[6] = noheart.copy()
    raw._data[8] = noisy.copy()

    # calculate PP quality
    raw, scores, times_out = peak_power(raw, time_window=time_window, threshold=0.1)

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 3)
    assert len(times_out) == 3

    # verify scores
    print(scores[:9, 1])
    from pprint import pprint

    print(raw.ch_names[:9])
    pprint([ann for ann in raw.annotations if ann["description"] == "BAD_PeakPower"])
    assert_allclose(scores[0:3, 1], 1.34, atol=0.01)  # group 0
    assert_allclose(scores[3:9, 1], 0.03, atol=0.01)  # groups 2 and 3
    assert_allclose(
        scores[3:6, 1], scores[6:9, 1], atol=0.01
    )  # groups 2 and 3 should be about the same

    # verify that BAD_PeakPower annotations exist for groups 2 and 3 in the 2nd window
    marks = find_annotations(
        raw,
        "BAD_PeakPower",
        [1],
        raw.ch_names[:num_channels],
        time_window,
    )

    expected = np.array([False] * 3 + [True] * 6)
    assert_array_equal(marks.ravel(), expected)


def test_sci_windowed_annotations_target_correct_channels() -> None:
    """Test that BAD_SCI annotations are assigned to the correct channels.

    _validate_nirs_info / _check_channels_ordered returns picks sorted
    alphabetically by channel name, which can differ from the row order
    in raw._data. This test verifies that annotations are added to the
    correct channels regardless of the order of the channel names.
    """
    sfreq = 10.0
    n_samples = 400  # 40 s at 10 Hz → 4 windows of 10 s

    # Channels (both wavelengths and S-D numbers) stored in NON-alphabetical order
    ch_names = [
        "S2_D1 760",
        "S1_D1 850",
        "S1_D1 760",
        "S2_D1 850",
    ]
    ch_types = ["fnirs_od"] * 4

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        # Wavelength must be stored in loc[9] (required by _validate_nirs_info).
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n_samples)

    # S1_D1 (raw._data rows 2, 1): identical  → SCI = +1.0  (GOOD, above threshold)
    # S2_D1 (raw._data rows 0, 3): anti-corr  → SCI = −1.0  (BAD,  below threshold)
    data = np.array([signal, signal, signal, -signal], dtype=float)

    raw = mne.io.RawArray(data, info)
    raw_out, _, _ = scalp_coupling_index_windowed(raw, time_window=10, threshold=0.7)

    bad_annotations = [
        ann for ann in raw_out.annotations if ann["description"] == "BAD_SCI"
    ]
    bad_channels = {ann["ch_names"] for ann in bad_annotations}
    assert bad_channels == {("S2_D1 760", "S2_D1 850")}, (
        "BAD_SCI annotations were assigned to the wrong channels."
    )


def test_pp_windowed_annotations_target_correct_channels() -> None:
    """Test that BAD_PeakPower annotations are assigned to the correct channels.

    _validate_nirs_info / _check_channels_ordered returns picks sorted
    alphabetically by channel name, which can differ from the row order
    in raw._data. This test verifies that annotations are added to the
    correct channels regardless of the order of the channel names.
    """
    sfreq = 10.0
    n_samples = 400  # 40 s at 10 Hz → 4 windows of 10 s

    # Channels (both wavelengths and S-D numbers) stored in NON-alphabetical order
    ch_names = [
        "S2_D1 760",
        "S1_D1 850",
        "S1_D1 760",
        "S2_D1 850",
    ]
    ch_types = ["fnirs_od"] * 4

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        # Wavelength must be stored in loc[9] (required by _validate_nirs_info).
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])

    t = np.arange(n_samples) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t)  # base "heartbeat", 1 Hz
    noheart = np.sin(2 * np.pi * 0.1 * t) + np.sin(
        2 * np.pi * 3 * t
    )  # 0.1 Hz + 3 Hz, not in the 0.5-2.5 Hz PP band

    # S1_D1 (raw._data rows 2, 1): identical  → high PP  (GOOD, above threshold)
    # S2_D1 (raw._data rows 0, 3): no heartbeat  → low PP  (BAD,  below threshold)
    data = np.array([signal, signal, signal, noheart], dtype=float)

    raw = mne.io.RawArray(data, info)
    raw_out, _, _ = peak_power(raw, time_window=10, threshold=0.1)

    bad_annotations = [
        ann for ann in raw_out.annotations if ann["description"] == "BAD_PeakPower"
    ]
    bad_channels = {ann["ch_names"] for ann in bad_annotations}
    assert bad_channels == {("S2_D1 760", "S2_D1 850")}, (
        "BAD_PeakPower annotations were assigned to the wrong channels."
    )
